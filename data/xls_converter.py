import pandas as pd

df = pd.read_excel('QuantResearch-CaseStudy-MarketData-25.xlsx', header=None)

# Extract header information
instruments = df.iloc[0]  # First row: instrument names
price_types = df.iloc[1]  # Second row: Ask Price, Bid Price, Mid Price
metrics = df.iloc[2]      # Third row: PX_ASK, PX_BID, PX_MID

# Create new header row
new_headers = ['Date']

# Build column names
for i in range(1, len(instruments)):
    instrument = instruments[i]
    price_type = price_types[i]
    metric = metrics[i]
    
    # If instrument is NaN, use the last non-NaN instrument
    if pd.isna(instrument):
        instrument = new_headers[-1].split('_')[0]
    
    # Create column name
    if 'Ask' in str(price_type):
        suffix = 'ASK'
    elif 'Bid' in str(price_type):
        suffix = 'BID'
    elif 'Mid' in str(price_type):
        suffix = 'MID'
    else:
        suffix = str(metric).split('_')[-1]
    
    col_name = f"{instrument}_{suffix}"
    new_headers.append(col_name)

# Set new headers and remove the first 3 rows
df.columns = new_headers
df = df.iloc[3:].reset_index(drop=True)

# Ensure proper date formatting
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Convert numeric columns
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.columns = df.columns.str.replace(' rate', '')
df.columns = df.columns.str.replace(' implied vol', '_VOL')
df.columns = df.columns.str.replace('Δ', 'DELTA')
df.columns = df.columns.str.replace(' ', '_')

# Save to CSV
df.to_csv('market_data.csv', index=False)
