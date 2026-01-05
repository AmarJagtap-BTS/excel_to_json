import argparse
import pandas as pd
import json
import os
import sys
import glob

import re
import html

def main():
    parser = argparse.ArgumentParser(description='Parse Excel file(s) to JSON.')
    parser.add_argument('--input', '-i', help='Path to input Excel file or folder. If folder, processes all Excel files.', default='input')
    parser.add_argument('--output', '-o', help='Path to the output JSON file. If not provided, prints to stdout.')
    parser.add_argument('--sheet', '-s', help='Sheet name to parse. Defaults to the first sheet.', default=0)
    
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    sheet = args.sheet

    # Handle sheet name vs index
    try:
        sheet = int(sheet)
    except ValueError:
        pass  # It's a string name

    # Find Excel files
    excel_files = []
    
    if os.path.isdir(input_path):
        # Get all Excel files from directory
        excel_files = glob.glob(os.path.join(input_path, '*.xlsx')) + glob.glob(os.path.join(input_path, '*.xls'))
        if not excel_files:
            print(f"Error: No Excel files found in directory: {input_path}", file=sys.stderr)
            sys.exit(1)
        # Use the first Excel file found
        input_file = excel_files[0]
        print(f"Found Excel file: {os.path.basename(input_file)}")
    elif os.path.isfile(input_path):
        input_file = input_path
    else:
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_excel(input_file, sheet_name=sheet)

        # Helper function to clean text
        def clean_text(text):
            if pd.isna(text):
                return None
            t = str(text)
            # Remove HTML tags
            t = re.sub(r'<[^>]+>', '', t)
            # Decode HTML entities
            t = html.unescape(t)
            
            t = t.replace('\\\\n', '\n').replace('\\n', '\n')
            t = t.replace('\\\\ul', '') # Remove literal bullet markers or replace with symbol if preferred
            return t.strip()

        # Helper to parse long description into list
        def parse_description(text):
            if pd.isna(text):
                return None
            # first clean it
            t = clean_text(text)
            if not t:
                return []
            # Split by newline and filter empty
            lines = [line.strip() for line in t.split('\n') if line.strip()]
            return lines

        # clean column names (optional but good practice)
        df.columns = df.columns.str.strip()

        # Specific column handling
        if 'HSN/SC Code' in df.columns:
            # Convert to numeric, coercing errors to NaN, then to nullable Int64
            df['HSN/SC Code'] = pd.to_numeric(df['HSN/SC Code'], errors='coerce').astype('Int64')

        # Clean string columns
        string_cols = ['Terms & Conditions', 'Product Short Description', 'Product Title', 'Brand Name']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

        # Parse Description into list
        if 'Product Long Description' in df.columns:
             df['Product Long Description'] = df['Product Long Description'].apply(parse_description)

        # Convert to list of dicts manually to handle nan/Int64 serialization gracefully if needed, 
        # but to_json with orient='records' usually handles basic types well. 
        # However, 'Int64' might serialize as integer which is what we want.
        
        # Add primary key (auto-incrementing ID starting from 1)
        df.insert(0, 'id', range(1, len(df) + 1))
        
        # Convert to JSON records
        # force_ascii=False to keep unicode characters like rupee symbol
        json_data = df.to_json(orient='records', date_format='iso', indent=4, force_ascii=False)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
            print(f"Successfully converted '{input_file}' to '{output_path}'")
        else:
            print(json_data)

    except Exception as e:
        print(f"Error parsing Excel file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
