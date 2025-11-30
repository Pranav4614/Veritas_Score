"""Updated app.py: adds local anchoring ledger, ECDSA signing, key management, and API/UI endpoints.
This file replaces the previous Flask application with a version that:
- Parses PhonePe-style PDFs into transactions
- Computes a Veritas reputation score
- Anchors PDF + transactions to a local chain (chain.json) signed with ECDSA
- Exposes web UI and JSON APIs
"""

import os
import io
import re
import json
import hashlib
from time import time
from base64 import b64encode, b64decode
from datetime import datetime

from flask import Flask, request, render_template_string, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import pdfplumber
import pandas as pd
import numpy as np
from ecdsa import SigningKey, VerifyingKey, SECP256k1, BadSignatureError

# ---------------- Configuration ----------------
UPLOAD_FOLDER = 'uploads'
KEY_FOLDER = 'keys'
CHAIN_FILE = 'chain.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KEY_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}
SCORE_MIN = 0
SCORE_MAX = 1000

app = Flask(__name__)
CORS(app)

# ---------------- Utility & Scoring Model ----------------
def min_max_scale(series, new_min=0, new_max=1):
    s = pd.Series(series).astype(float)
    if s.empty or s.min() == s.max():
        return pd.Series((new_min + new_max) / 2, index=s.index)
    return (s - s.min()) / (s.max() - s.min()) * (new_max - new_min) + new_min

class VeritasScoreCalculator:
    def __init__(self):
        self.WEIGHT_NET_FLOW = 0.4
        self.WEIGHT_AVG_CREDIT = 0.3
        self.WEIGHT_SPENDING_DISCIPLINE = 0.3

    def calculate_reputation_index(self, transactions_df):
        df = transactions_df.copy()
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['type'] = df['type'].str.lower().fillna('debit')

        df_credit = df[df['type'] == 'credit']
        df_debit = df[df['type'] == 'debit']

        credit_summary = df_credit.groupby('user_id').agg(
            total_credit=('amount', 'sum'),
            avg_credit_amount=('amount', 'mean'),
            credit_txn_count=('txn_id', 'count')
        ).fillna(0)

        debit_summary = df_debit.groupby('user_id').agg(
            total_debit=('amount', 'sum'),
            debit_txn_count=('txn_id', 'count')
        ).fillna(0)

        user_features = credit_summary.merge(debit_summary, left_index=True, right_index=True, how='outer').fillna(0)

        user_features['net_financial_flow'] = user_features['total_credit'] - user_features['total_debit']
        user_features['avg_credit_amount_norm'] = user_features['avg_credit_amount']

        user_features['spending_discipline_ratio'] = np.where(
            user_features['total_credit'] == 0,
            1.0,
            (user_features['total_debit'] / user_features['total_credit'])
        )
        user_features['spending_discipline_ratio'] = user_features['spending_discipline_ratio'].clip(upper=1.0)

        user_features['scaled_net_financial_flow'] = min_max_scale(user_features['net_financial_flow'], 0, 1)
        user_features['scaled_avg_credit_amount'] = min_max_scale(user_features['avg_credit_amount_norm'], 0, 1)
        user_features['scaled_spending_disc'] = 1 - min_max_scale(user_features['spending_discipline_ratio'], 0, 1)

        user_features['reputation_score'] = (
            user_features['scaled_net_financial_flow'] * self.WEIGHT_NET_FLOW +
            user_features['scaled_avg_credit_amount'] * self.WEIGHT_AVG_CREDIT +
            user_features['scaled_spending_disc'] * self.WEIGHT_SPENDING_DISCIPLINE
        )

        user_features['reputation_index'] = min_max_scale(user_features['reputation_score'], SCORE_MIN, SCORE_MAX).round(2)

        user_features['total_credit'] = user_features['total_credit']
        user_features['total_debit'] = user_features['total_debit']
        user_features['net_financial_flow'] = user_features['net_financial_flow']

        user_features = user_features.reset_index().rename(columns={'index':'user_id'})
        return user_features

    def _choose_rating(self, score):
        if score >= 800:
            return 'EXCELLENT'
        elif score >= 600:
            return 'GOOD'
        elif score >= 400:
            return 'FAIR'
        elif score >= 200:
            return 'NEEDS IMPROVEMENT'
        else:
            return 'POOR'

# ---------------- PhonePe PDF parser (heuristic) ----------------
date_line_pattern = re.compile(r'([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})')
time_line_pattern = re.compile(r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))')
amount_pattern = re.compile(r'₹\s?([\d,]+(?:\.\d{1,2})?)')
txn_id_pattern = re.compile(r'(Transaction ID|Transaction\s*Id|Transaction id)\s*[:\s]*([A-Za-z0-9\-]+)', re.IGNORECASE)
credit_word = re.compile(r'\bCREDIT\b', re.IGNORECASE)
debit_word = re.compile(r'\bDEBIT\b', re.IGNORECASE)
received_from = re.compile(r'Received from\s*(.+)', re.IGNORECASE)
paid_to = re.compile(r'Paid to\s*(.+)', re.IGNORECASE)

def parse_phonepe_pdf(file_bytes):
    text_lines = []
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                lines = [ln.strip() for ln in page_text.split('\n') if ln.strip()!='']
                text_lines.extend(lines)

    transactions = []
    i = 0
    last_date = None
    while i < len(text_lines):
        line = text_lines[i]
        date_match = date_line_pattern.search(line)
        if date_match:
            last_date = date_match.group(1).strip()
            i += 1
            continue

        time_match = time_line_pattern.search(line)
        if time_match and last_date:
            time_str = time_match.group(1).upper()
            timestamp = None
            try:
                dt_str = f"{last_date} {time_str}"
                timestamp = datetime.strptime(dt_str, '%b %d, %Y %I:%M %p')
            except Exception:
                try:
                    timestamp = datetime.strptime(dt_str, '%B %d, %Y %I:%M %p')
                except Exception:
                    timestamp = None

            block = ''
            for j in range(1,5):
                if i + j < len(text_lines):
                    block += ' ' + text_lines[i + j]

            amount_m = amount_pattern.search(block)
            txn_m = txn_id_pattern.search(block)
            credit_m = credit_word.search(block)
            debit_m = debit_word.search(block)
            merchant = None
            cat = None

            rcv = received_from.search(block)
            pto = paid_to.search(block)
            if rcv:
                merchant = rcv.group(1).split('Transaction')[0].strip()
                cat = 'income'
                tx_type = 'credit'
            elif pto:
                merchant = pto.group(1).split('Transaction')[0].strip()
                cat = 'expense'
                tx_type = 'debit'
            else:
                if 'Received from' in block:
                    rem = block.split('Received from')[-1].split('Transaction')[0].strip()
                    merchant = rem
                    tx_type = 'credit'
                    cat = 'income'
                elif 'Paid to' in block:
                    rem = block.split('Paid to')[-1].split('Transaction')[0].strip()
                    merchant = rem
                    tx_type = 'debit'
                    cat = 'expense'
                else:
                    if credit_m:
                        tx_type = 'credit'
                        cat = 'income'
                    elif debit_m:
                        tx_type = 'debit'
                        cat = 'expense'
                    else:
                        tx_type = 'debit'
                        cat = 'other'

            amount = 0.0
            if amount_m:
                amt_str = amount_m.group(1).replace(',', '')
                try:
                    amount = float(amt_str)
                except:
                    amount = 0.0

            txn_id = None
            if txn_m:
                txn_id = txn_m.group(2).strip()
            else:
                for j in range(1,6):
                    if i + j < len(text_lines):
                        line_j = text_lines[i+j]
                        m = re.search(r'\bT\d{6,}\S*|NB\d{6,}\S*|UTR No\.\s*([A-Za-z0-9\-]+)', line_j)
                        if m:
                            txn_id = m.group(0).strip()
                            break

            transactions.append({
                'user_id': 1,
                'txn_id': txn_id if txn_id else f"txn_{len(transactions)+1}",
                'timestamp': timestamp if timestamp else pd.NaT,
                'amount': amount,
                'type': tx_type,
                'category': cat if cat else '',
                'merchant': merchant if merchant else '',
            })
            i += 4
            continue

        amt_inline = amount_pattern.search(line)
        if amt_inline and ('Received' in line or 'Paid' in line or 'CREDIT' in line.upper() or 'DEBIT' in line.upper()):
            amount = float(amt_inline.group(1).replace(',', ''))
            tx_type = 'credit' if 'CREDIT' in line.upper() or 'Received' in line else 'debit'
            merchant = ''
            rcv = received_from.search(line)
            pto = paid_to.search(line)
            if rcv:
                merchant = rcv.group(1).split('Transaction')[0].strip()
            elif pto:
                merchant = pto.group(1).split('Transaction')[0].strip()
            txn_m = txn_id_pattern.search(line)
            txn_id = txn_m.group(2) if txn_m else f"txn_{len(transactions)+1}"
            transactions.append({
                'user_id':1, 'txn_id': txn_id, 'timestamp': pd.NaT,
                'amount': amount, 'type': tx_type, 'category': '', 'merchant': merchant
            })
        i += 1

    if len(transactions) == 0:
        return pd.DataFrame(columns=['user_id','txn_id','timestamp','amount','type','category','merchant'])

    df = pd.DataFrame(transactions)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['txn_id'] = df['txn_id'].fillna('').astype(str)
    for idx in df[df['txn_id']==''].index:
        df.at[idx,'txn_id'] = f"txn_{idx+1}"
    return df

# ---------------- Local blockchain-like ledger ----------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _load_chain():
    if not os.path.exists(CHAIN_FILE):
        return []
    with open(CHAIN_FILE, 'r') as f:
        try:
            return json.load(f)
        except:
            return []

def _save_chain(chain):
    with open(CHAIN_FILE, 'w') as f:
        json.dump(chain, f, indent=2, sort_keys=True)

def generate_keypair(save_private_to=None):
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.verifying_key
    if save_private_to:
        with open(save_private_to, 'wb') as f:
            f.write(sk.to_pem())
        try:
            os.chmod(save_private_to, 0o600)
        except Exception:
            pass
        # also write public key PEM for easy distribution
        try:
            pub_path = os.path.join(KEY_FOLDER, 'veritas_public.pem')
            with open(pub_path, 'wb') as pf:
                pf.write(vk.to_pem())
        except Exception:
            pass
    return sk, vk

def load_private_key(path):
    with open(path, 'rb') as f:
        return SigningKey.from_pem(f.read())

def create_block(file_hash, txns_hash, signature_b64, public_key_hex, meta=None):
    chain = _load_chain()
    previous_hash = chain[-1]['block_hash'] if chain else ''
    timestamp = int(time())
    block = {
        'index': len(chain) + 1,
        'timestamp': timestamp,
        'previous_hash': previous_hash,
        'file_hash': file_hash,
        'txns_hash': txns_hash,
        'signature': signature_b64,
        'public_key': public_key_hex,
        'meta': meta or {}
    }
    block_contents = json.dumps(block, sort_keys=True, separators=(',', ':')).encode('utf-8')
    block_hash = hashlib.sha256(block_contents).hexdigest()
    block['block_hash'] = block_hash
    chain.append(block)
    _save_chain(chain)
    return len(chain)-1, block

def verify_block_signature(block):
    try:
        vk_hex = block['public_key']
        vk_bytes = bytes.fromhex(vk_hex)
        vk = VerifyingKey.from_string(vk_bytes, curve=SECP256k1)
        message = block['file_hash'].encode('utf-8')
        signature_b64 = block['signature']
        sig_bytes = b64decode(signature_b64)
        vk.verify(sig_bytes, message)
        return True
    except BadSignatureError:
        return False
    except Exception as e:
        print("verify error:", e)
        return False

def anchor_to_chain(pdf_bytes: bytes, transactions_df: pd.DataFrame, private_key_path=None, meta=None):
    file_hash = sha256_bytes(pdf_bytes)
    txns_json = json.dumps(transactions_df.to_dict(orient='records'), sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
    txns_hash = hashlib.sha256(txns_json).hexdigest()

    key_path = private_key_path or os.path.join(KEY_FOLDER, 'veritas_private.pem')
    if not os.path.exists(key_path):
        sk, vk = generate_keypair(save_private_to=key_path)
    else:
        sk = load_private_key(key_path)
        vk = sk.verifying_key

    signature_b64 = b64encode(sk.sign(file_hash.encode('utf-8'))).decode('utf-8')
    public_key_hex = vk.to_string().hex()

    block_index, block = create_block(file_hash, txns_hash, signature_b64, public_key_hex, meta=meta)
    return block

# ---------------- HTML templates ----------------
UPLOAD_PAGE = """
<!doctype html>
<html>
<head><title>Veritas Score — Upload PhonePe PDF</title></head>
<body>
<h2>Upload PhonePe Statement PDF</h2>
<form method=post enctype=multipart/form-data action="{{ url_for('calculate_score') }}">
  <input type=file name=file accept=".pdf" required>
  <br><br>
  <label>User ID (optional): <input type="text" name="user_id" value="1"></label>
  <br><br>
  <button type="submit">Calculate Score</button>
</form>
<p style="color:gray;margin-top:20px;">Parser tuned for PhonePe-like statements (date line, time, CREDIT/DEBIT, ₹Amount, Received from / Paid to, Transaction ID).</p>
</body>
</html>
"""

RESULT_PAGE = """
<!doctype html>
<html>
<head><title>Veritas Score Result</title></head>
<body>
<h2>Veritas Score Result</h2>
<p><strong>User ID:</strong> {{ user_id }}</p>
<p><strong>Score:</strong> {{ score }} ({{ rating }})</p>
<p><strong>Total credit:</strong> ₹{{ total_credit }}</p>
<p><strong>Total debit:</strong> ₹{{ total_debit }}</p>
<p><strong>Net flow:</strong> ₹{{ net_flow }}</p>
<p><strong>Transaction count:</strong> {{ txn_count }}</p>

<h3>Anchoring (local ledger)</h3>
<p><strong>Block index:</strong> {{ block.index }}</p>
<p><strong>Block hash:</strong> {{ block.block_hash }}</p>
<p><strong>File hash (sha256):</strong> {{ block.file_hash }}</p>
<p><strong>Transactions hash (sha256):</strong> {{ block.txns_hash }}</p>
<p><strong>Signature (base64):</strong> {{ block.signature }}</p>
<p><strong>Public key (hex):</strong> {{ block.public_key }}</p>
<p><a href="{{ url_for('verify_block_ui', block_index=block.index) }}">Verify block signature</a></p>

<h3>Transactions (first 200)</h3>
<div style="max-height:400px; overflow:auto; border:1px solid #ddd; padding:8px;">
  {{ table_html | safe }}
</div>

<br>
<a href="{{ url_for('index') }}">Upload another file</a>
</body>
</html>
"""

VERIFY_PAGE = """
<!doctype html>
<html>
<head><title>Verify Block</title></head>
<body>
<h2>Verify Block #{{ block_index }}</h2>
{% if not found %}
  <p>Block not found.</p>
{% else %}
  <p><strong>Block hash:</strong> {{ block.block_hash }}</p>
  <p><strong>Signature valid:</strong> {{ sig_ok }}</p>
  <pre>{{ block | tojson(indent=2) }}</pre>
{% endif %}
<br>
<a href="{{ url_for('index') }}">Back</a>
</body>
</html>
"""

# ---------------- Flask routes ----------------
@app.route('/', methods=['GET'])
def index():
    return render_template_string(UPLOAD_PAGE)

@app.route('/calculate', methods=['POST'])
def calculate_score():
    if 'file' not in request.files:
        return "No file provided", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    if not ('.' in file.filename and file.filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS):
        return "Only PDF files are allowed", 400

    user_id = request.form.get('user_id', '1')
    try:
        user_id_int = int(user_id)
    except:
        user_id_int = 1

    filename = secure_filename(file.filename)
    file_bytes = file.read()

    # save uploaded file for record (optional)
    save_path = os.path.join(UPLOAD_FOLDER, f"{int(time())}_{filename}")
    try:
        with open(save_path, 'wb') as f:
            f.write(file_bytes)
    except Exception:
        pass

    # parse pdf
    try:
        transactions_df = parse_phonepe_pdf(io.BytesIO(file_bytes))
    except Exception as e:
        return f"Error parsing PDF: {e}", 500

    if transactions_df.empty:
        return "No transactions parsed from PDF — parser may need tuning for this statement format.", 400

    transactions_df['user_id'] = user_id_int

    if 'txn_id' not in transactions_df.columns:
        transactions_df['txn_id'] = [f"txn_{i+1}" for i in range(len(transactions_df))]

    # score
    calc = VeritasScoreCalculator()
    user_features = calc.calculate_reputation_index(transactions_df)

    row = user_features[user_features['user_id'] == user_id_int]
    if row.empty:
        total_credit = transactions_df[transactions_df['type']=='credit']['amount'].sum()
        total_debit = transactions_df[transactions_df['type']=='debit']['amount'].sum()
        net_flow = total_credit - total_debit
        tmp_score = float(min(max((net_flow / (abs(net_flow) + 1)) * 500 + 500, 0), 1000))
        score_val = round(tmp_score,2)
        rating = calc._choose_rating(score_val)
    else:
        score_val = float(row['reputation_index'].iloc[0])
        total_credit = float(row['total_credit'].iloc[0])
        total_debit = float(row['total_debit'].iloc[0])
        net_flow = float(row['net_financial_flow'].iloc[0])
        rating = calc._choose_rating(score_val)

    # anchor to local chain
    meta = {'uploaded_filename': filename, 'user_id': user_id_int, 'score': score_val}
    block = anchor_to_chain(file_bytes, transactions_df, meta=meta)

    table_html = transactions_df.head(200).to_html(index=False, classes='table table-sm')

    return render_template_string(RESULT_PAGE,
                                  user_id=user_id_int,
                                  score=score_val,
                                  rating=rating,
                                  total_credit=round(total_credit,2),
                                  total_debit=round(total_debit,2),
                                  net_flow=round(net_flow,2),
                                  txn_count=len(transactions_df),
                                  table_html=table_html,
                                  block=block)

# API route to anchor file only (returns block)
@app.route('/api/anchor', methods=['POST'])
def api_anchor():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400
    file = request.files['file']
    file_bytes = file.read()
    try:
        transactions_df = parse_phonepe_pdf(io.BytesIO(file_bytes))
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error parsing PDF: {str(e)}'}), 500
    meta = {'uploaded_filename': secure_filename(file.filename), 'user_id': request.form.get('user_id','1')}
    block = anchor_to_chain(file_bytes, transactions_df, meta=meta)
    return jsonify({'success': True, 'block': block}), 200

@app.route('/api/chain', methods=['GET'])
def api_chain():
    chain = _load_chain()
    return jsonify({'length': len(chain), 'chain': chain}), 200

@app.route('/api/verify-block/<int:block_index>', methods=['GET'])
def api_verify_block(block_index):
    chain = _load_chain()
    if block_index <= 0 or block_index > len(chain):
        return jsonify({'success': False, 'message': 'invalid block index'}), 404
    block = chain[block_index-1]
    sig_ok = verify_block_signature(block)
    return jsonify({'success': True, 'block': block, 'signature_valid': sig_ok}), 200

# UI verify
@app.route('/verify/<int:block_index>', methods=['GET'])
def verify_block_ui(block_index):
    chain = _load_chain()
    if block_index <= 0 or block_index > len(chain):
        return render_template_string(VERIFY_PAGE, block_index=block_index, found=False)
    block = chain[block_index-1]
    sig_ok = verify_block_signature(block)
    return render_template_string(VERIFY_PAGE, block_index=block_index, found=True, block=block, sig_ok=sig_ok)

# ---------------- Download endpoints ----------------
@app.route('/download/uploaded/<path:filename>', methods=['GET'])
def download_uploaded(filename):
    # secure the filename and serve from uploads directory
    safe_name = secure_filename(filename)
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    if not os.path.exists(file_path):
        return ("File not found", 404)
    return send_from_directory(UPLOAD_FOLDER, safe_name, as_attachment=True)


@app.route('/download/chain', methods=['GET'])
def download_chain():
    if not os.path.exists(CHAIN_FILE):
        return ("No chain file found", 404)
    return send_file(CHAIN_FILE, mimetype='application/json', as_attachment=True, download_name='chain.json')


@app.route('/download/public-key', methods=['GET'])
def download_public_key():
    pub_path = os.path.join(KEY_FOLDER, 'veritas_public.pem')
    # If public key not present but private exists, generate public PEM
    priv_path = os.path.join(KEY_FOLDER, 'veritas_private.pem')
    if not os.path.exists(pub_path):
        if os.path.exists(priv_path):
            try:
                sk = load_private_key(priv_path)
                vk = sk.verifying_key
                with open(pub_path, 'wb') as pf:
                    pf.write(vk.to_pem())
            except Exception:
                pass
    if not os.path.exists(pub_path):
        return ("Public key not available", 404)
    return send_file(pub_path, mimetype='application/x-pem-file', as_attachment=True, download_name='veritas_public.pem')

# ---------------- Run ----------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
