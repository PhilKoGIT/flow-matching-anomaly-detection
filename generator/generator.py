

##################################################

#Generator for the SAP Fioneer business transaction dataset
#not my own work, received from SAP Fioneer

####################################################


import pandas as pd
import numpy as np
from faker import Faker
from datetime import date, timedelta
import random
import uuid
import os


NUM_CUSTOMERS = 4
START_DATE = date(2022, 1, 1)
END_DATE = date(2023, 12, 31)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "business_dataset.csv")

fake = Faker()
Faker.seed(0)
random.seed(0)
np.random.seed(0)

def generate_clean_company_name(faker_instance):
    """Generates a company name, rejecting those with commas."""
    while True:
        name = faker_instance.company()
        if ',' not in name:
            return name


def generate_transaction_series(
    bank_account_uuid, business_partner_name, series_start_date, series_end_date,
    ref_name, base_amount, amount_std_dev, day_of_month,
    ref_iban, ref_swift, pay_method, channel
):
    """Generates 'normal' monthly transactions with variable payment terms and growth trends"""
    transactions = []
    current_date = series_start_date.replace(day=1)
    invoice_counter = random.randint(1000, 5000)

    # Assign a payment term and growth rate per series
    term_days = random.choice([15, 30, 45, 60])
    monthly_growth = random.uniform(0.005, 0.02)
    month_idx = 0

    while current_date <= series_end_date:
        growth_type = random.choice(['none','progression','regression'])
        if growth_type == 'none':
            cur_base = base_amount
        elif growth_type == 'progression':
            cur_base = base_amount * ((1 + monthly_growth) ** month_idx)
        elif growth_type == 'regression':
            cur_base = base_amount * ((1 + monthly_growth) ** -month_idx)
        
        #Erzeugt eine zufällige Zahl aus einer Normalverteilung (Glockenkurve).
        transaction_amount = round(np.random.normal(cur_base, amount_std_dev), 2)

        inv_day = max(1, min(28, day_of_month))
        invoice_date = date(current_date.year, current_date.month, inv_day)

        # Add random payment date change +- 1 day
        delay = np.random.poisson(lam=1)
        payment_date = invoice_date + timedelta(days=term_days + delay)
        # If payment falls on a weekend, push to next business day
        if payment_date.weekday() >= 5:
            payment_date += timedelta(days=7 - payment_date.weekday())

        note_prefix = random.choice(["Invoice #", "Inv. #", "Payment for "])
        payment_note = (
            f"{note_prefix}INV-{invoice_date.year}-{invoice_date.month:02d}-"
            f"{invoice_counter} (Term {term_days}d)"
        )

        transactions.append({
            "bank_account_uuid": bank_account_uuid,
            "business_partner_name": business_partner_name,
            "date_post": payment_date.strftime('%Y%m%d'),
            "amount": transaction_amount,
            "currency": "USD",
            "ref_name": ref_name,
            "ref_bank": fake.country_code(),
            "paym_note": payment_note,
            "trns_type": "DEBIT",
            "pay_method": pay_method,
            "channel": channel,
            "ref_iban": ref_iban,
            "ref_swift": ref_swift,
            "anomaly_description": None
        })
        invoice_counter += 1
        month_idx += 1
        if current_date.month == 12:
            current_date = date(current_date.year + 1, 1, 1)
        else:
            current_date = date(current_date.year, current_date.month + 1, 1)

    return transactions


def generate_payroll_batch(customer_uuid, customer_name, target_date):
    payroll = []
    num_employees = random.randint(5, 50)
    for _ in range(num_employees):
        payroll.append({
            "bank_account_uuid": customer_uuid,
            "business_partner_name": customer_name,
            "date_post": target_date.strftime('%Y%m%d'),
            "amount": abs(round(np.random.normal(4500, 1500), 2)),
            "currency": "USD",
            "ref_name": fake.name(),
            "ref_bank": fake.country_code(),
            "paym_note": f"Salary for {target_date.strftime('%B %Y')}",
            "trns_type": "DEBIT",
            "pay_method": "ACH",
            "channel": "API",
            "ref_iban": fake.iban(),
            "ref_swift": fake.swift(length=11),
            "anomaly_description": None
        })
    return payroll

# Generate one-off payments only in Q1 & Q3 to avoid the LLM thinking there is a payment pattern
QUARTERS = [1, 3]

def generate_one_off_payments(customer_uuid, customer_name):
    one_offs = []
    for q in QUARTERS:
        for year in range(START_DATE.year, END_DATE.year + 1):
            count = random.randint(2, 5)
            for _ in range(count):
                start_m = 3*(q-1) + 1
                month = random.randint(start_m, start_m + 2)
                day = random.randint(1, 28)
                dt = date(year, month, day)
                if dt < START_DATE or dt > END_DATE:
                    continue
                note_desc = random.choice(["Consulting Services", "New Equipment Purchase", "Legal Fees"])
                one_offs.append({
                    "bank_account_uuid": customer_uuid,
                    "business_partner_name": customer_name,
                    "date_post": dt.strftime('%Y%m%d'),
                    "amount": round(random.uniform(10000, 75000), 2),
                    "currency": "USD",
                    "ref_name": generate_clean_company_name(fake),
                    "ref_bank": fake.country_code(),
                    "paym_note": f"{note_desc} - PO#{random.randint(2000,9999)}",
                    "trns_type": "DEBIT",
                    "pay_method": "WIRE",
                    "channel": "ONLINE_BANKING_PORTAL",
                    "ref_iban": fake.iban(),
                    "ref_swift": fake.swift(length=11),
                    "anomaly_description": None
                })
    return one_offs


def inject_amount_anomaly(transactions):
    if not transactions: return transactions
    idx = random.randint(0, len(transactions) - 1)
    original_amount = transactions[idx]['amount']
    spike_factor = random.choice([1.5, 1.8, 2.0])
    transactions[idx]['amount'] = round(original_amount * spike_factor, 2)
    transactions[idx]['anomaly_description'] = f"AMOUNT_ANOMALY: Amount changed from ~{original_amount:.0f}"
    return transactions

def inject_frequency_anomaly(transactions):
    if not transactions: return transactions
    idx = random.randint(0, len(transactions) - 1)
    rogue = transactions[idx].copy()
    orig = date.fromisoformat(f"{rogue['date_post'][:4]}-{rogue['date_post'][4:6]}-{rogue['date_post'][6:]}")
    new_day = (orig.day + random.randint(7,15))%28 + 1
    rogue['date_post'] = orig.replace(day=new_day).strftime('%Y%m%d')
    rogue['paym_note'] = "REPAYMENT-URGENT"
    rogue['anomaly_description'] = "FREQUENCY_ANOMALY: Duplicate monthly payment"
    transactions.insert(idx, rogue)
    return transactions



def inject_payee_anomaly(transactions):
    if not transactions: return transactions
    idx = random.randint(0, len(transactions)-1)
    orig = transactions[idx]['ref_name']
    transactions[idx]['ref_name'] = generate_clean_company_name(fake) + " Ltd."
    transactions[idx]['ref_bank'] = fake.country_code()
    transactions[idx]['ref_iban'] = fake.iban()
    transactions[idx]['ref_swift'] = fake.swift(length=11)
    transactions[idx]['anomaly_description'] = f"PAYEE_ANOMALY: Payee changed from {orig}"
    return transactions


def inject_timing_anomaly(transactions):
    if not transactions: return transactions
    idx = random.randint(0, len(transactions)-1)
    orig = date.fromisoformat(f"{transactions[idx]['date_post'][:4]}-{transactions[idx]['date_post'][4:6]}-{transactions[idx]['date_post'][6:]}")
    new_day = (orig.day + 10)%28 + 1
    transactions[idx]['date_post'] = orig.replace(day=new_day).strftime('%Y%m%d')
    transactions[idx]['anomaly_description'] = f"TIMING_ANOMALY: Payment date shifted from day ~{orig.day} to {new_day}"
    return transactions


def inject_channel_anomaly(transactions):
    if not transactions: return transactions
    idx = random.randint(0,len(transactions)-1)
    orig_m = transactions[idx]['pay_method']
    orig_c = transactions[idx]['channel']
    transactions[idx]['pay_method'] = 'CARD'
    transactions[idx]['channel'] = 'MOBILE_APP'
    transactions[idx]['anomaly_description'] = f"CHANNEL_ANOMALY: Payment made via CARD/MOBILE_APP instead of {orig_m}/{orig_c}"
    return transactions

### raus
def inject_iban_mismatch_anomaly(transactions):
    if not transactions: return transactions
    idx = random.randint(0,len(transactions)-1)
    orig = transactions[idx]['ref_iban']
    new = fake.iban()
    transactions[idx]['ref_iban'] = new
    transactions[idx]['anomaly_description'] = f"IBAN_MISMATCH_ANOMALY: IBAN changed from {orig[:10]}... to {new[:10]}..."
    return transactions


def inject_subtle_payee_and_iban_anomaly(transactions):
    if not transactions: return transactions
    idx = random.randint(0,len(transactions)-1)
    orig = transactions[idx]['ref_name']
    if "Solutions" in orig:
        new_name = orig.replace("Solutions","Systems")
    elif "Cloud" in orig:
        new_name = orig.replace("Cloud","Core")
    else:
        new_name = orig.split(' ')[0] + " Ventures"
    transactions[idx]['ref_name'] = new_name
    transactions[idx]['ref_iban'] = fake.iban()
    transactions[idx]['anomaly_description'] = f"SUBTLE_PAYEE_MISMATCH: Payee changed from '{orig}' to '{new_name}'"
    return transactions


if __name__ == "__main__":
    all_transactions = []
    anomaly_injectors = [
        inject_amount_anomaly, inject_frequency_anomaly, 
        inject_payee_anomaly,
        inject_timing_anomaly, inject_channel_anomaly,
        inject_iban_mismatch_anomaly, 
        inject_subtle_payee_and_iban_anomaly
    ]

    print(f"Generating comprehensive transaction data for {NUM_CUSTOMERS} customers...")

    for i in range(NUM_CUSTOMERS):
        customer_uuid = str(uuid.uuid4()).replace('-', '').upper()
        customer_name = generate_clean_company_name(fake)
        print(f"  - ({i+1}/{NUM_CUSTOMERS}) Generating data for {customer_name}")

        # Create a clean series of transactions
        clean_series_ref = generate_clean_company_name(fake) + " Solutions"
        clean_series = generate_transaction_series(
            bank_account_uuid=customer_uuid,
            business_partner_name=customer_name,
            series_start_date=START_DATE,
            series_end_date=END_DATE,
            ref_name=clean_series_ref,
            base_amount=random.uniform(50000, 200000),
            amount_std_dev=5000,
            day_of_month=random.choice([1, 5]),
            ref_iban=fake.iban(),
            ref_swift=fake.swift(length=11),
            pay_method='ACH',
            channel='API',
            #day_jitter=1
        )
        # Inject an anomaly
        how_many_anomalies = random.randint(0, 20)
        which_anomaly = random.randint(0, len(anomaly_injectors)-1)
        for x in range(how_many_anomalies):
            clean_series = anomaly_injectors[which_anomaly](clean_series)
        all_transactions.extend(clean_series)

        # Add noise to the data: 5-10 additional series
        for _ in range(random.randint(5, 10)):
            bg_ref = generate_clean_company_name(fake) + " Logistics"
            bg_series = generate_transaction_series(
                bank_account_uuid=customer_uuid,
                business_partner_name=customer_name,
                series_start_date=START_DATE,
                series_end_date=END_DATE,
                ref_name=bg_ref,
                base_amount=random.uniform(500, 5000),
                amount_std_dev=50,
                day_of_month=random.choice([15, 20, 25]),
                ref_iban=fake.iban(),
                ref_swift=fake.swift(length=11),
                pay_method='WIRE',
                channel='ONLINE_BANKING_PORTAL',
                #day_jitter=1
            )
            all_transactions.extend(bg_series)


        # Add one-off payments in Q1 & Q3
        all_transactions.extend(generate_one_off_payments(customer_uuid, customer_name))

    df = pd.DataFrame(all_transactions)
    df = df.sort_values(by="date_post").sample(frac=1).reset_index(drop=True)
    column_order = [
        "bank_account_uuid", "business_partner_name", "date_post", "amount", "currency",
        "ref_name", "ref_iban", "ref_swift", "ref_bank", "paym_note", "trns_type",
        "pay_method", "channel", "anomaly_description"
    ]
    df = df[column_order]
    df.to_csv(OUTPUT_FILENAME, index=False)

    print(f"\nSuccessfully generated COMPREHENSIVE dataset with {len(df)} transactions.")
    print(f"Saved to '{OUTPUT_FILENAME}'")
    df.head(10)  # Display first 10 rows for verification

    pd.set_option('display.max_columns', None)  # Zeigt alle Spalten
    pd.set_option('display.width', None)        # 
    print(df.head(10))  # Display first 10 rows for verification