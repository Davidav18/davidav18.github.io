import pandas as pd
import os
import time
import vertexai
from vertexai.generative_models import GenerativeModel

# --- CONFIGURATION ---
PROJECT_ID = "++++++++++++"
LOCATION = "us-central1"
files_to_process = ['split_file_1.csv']
BATCH_SIZE = 2000   # Adjust according to Gemini's limits and your use case

# --- IMPROVED PROMPT ---
prompt_instruction = """
# Role:
Act as an expert software analyst. I will provide you with a list of software names, fonts, packages, etc.
Your task is to eliminate duplicates and consolidate this list very specifically: if multiple entries refer to the same base software, you should unify them under a single representative or generic name. Remove version numbers, architecture details (x64, x86), dates, and specific descriptors like 'Runtime', 'Update', 'Agent', or 'Edition'.

# Output Rules
Return the answer **only** as a CSV file **with two columns** named "Original" and "Consolidated", where:
- The "Original" column contains the original input value.
- The "Consolidated" column contains the reduced or grouped name.
- Do NOT include any explanatory text before or after, only the CSV file lines.
- If the name starts with `lib` and is a Linux package (e.g., libnspr4, libnss3, libxcomposite1, libxinerama1, etc.), its consolidated value or generic should be "System Library".
- Anything related to Python (including libraries and packages) should be consolidated as "Python Library".
- Anything related to fonts or “font” (including libraries, packages, and files) should be consolidated as "Text Font".
- If it is a DNS package or service like bind9, bind9-host, isc-dhcp-server, its consolidated value should be "Network Service".
- If it is a printer driver, controller, or package, consolidate as "Printer Driver".
- If the software is a web browser, consolidate as "Web Browser".
- If it is a compression utility or tool (zip, 7zip, gzip, bzip2, etc.), consolidate as "Compression Tool".
- If it is a text editor utility or package (vim, nano, gedit, notepad++), consolidate as "Text Editor".
- If it is antivirus or security software, consolidate as "Security Software".
- If it is a database management system (MySQL, PostgreSQL, Oracle, etc.), consolidate as "Database System".
- If it is a remote management or remote access tool, consolidate as "Remote Access Tool".
- If it is a virtualization tool (VMware, VirtualBox, etc.), consolidate as "Virtualization Tool".
- If it is an operating system or main operating system component, consolidate as "Operating System Component".
- If it does not fit any previous category, follow the consolidation criteria by software family, as in the examples.

# Output Format
Example of output format:
Original,Consolidated
Microsoft Visual C++ 2015-2022 Redistributable,Microsoft Visual C++
Microsoft Visual C++ 2022 x86 Minimum Runtime,Microsoft Visual C++
Microsoft Visual C++ 2022 x64 Minimum Runtime,Microsoft Visual C++
Windows 7 WDK Header and Libs,WDK Header and Libs
Windows 8 WDK Header and Libs,WDK Header and Libs
Windows 8.1 WDK Header and Libs,WDK Header and Libs
Windows Driver Kit ARM Additions,Windows Driver Kit
Windows Driver Kit ARM Binaries,Windows Driver Kit
Windows Driver Kit ARM Headers and Libs,Windows Driver Kit
Windows Driver Kit Binaries,Windows Driver Kit
libnspr4,System Library
libnss3,System Library
libxcomposite1,System Library
numpy,Python Library
Open Sans,Text Font
bind9,Network Service

Here is the list of data to process:
"""

# --- SCRIPT START ---
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-pro")
    print("✅ Connected to Google Gemini (Vertex AI) API successfully.")
    
except Exception as e:
    print(f"❌ ERROR: Could not initialize Vertex AI. Details: {e}")
    exit()

for file in files_to_process:
    print("-" * 40)
    print(f"Processing file: {file}")
    try:
        df = pd.read_csv(file)
        first_column_series = df.iloc[:, 0].dropna().astype(str).reset_index(drop=True)
        total = len(first_column_series)
        batch_results = []
        failed_batches = []

        for i in range(0, total, BATCH_SIZE):
            batch = first_column_series[i:i+BATCH_SIZE]
            text_to_send = '\n'.join(batch)
            full_prompt = prompt_instruction + "\n" + text_to_send
            batch_num = i // BATCH_SIZE + 1
            print(f"   -> Sending to Gemini... Batch {batch_num} ({i+1} to {min(i+BATCH_SIZE, total)})")
            start_time = time.time()
            try:
                response = model.generate_content(full_prompt)
                if hasattr(response, "text") and response.text:
                    ai_response = response.text.strip()
                else:
                    print("   ❌ Gemini response has no `.text` field or is empty.")
                    if hasattr(response, "candidates"):
                        print("   -> Response in 'candidates':", response.candidates)
                    raise Exception("Unexpected Gemini response, check previous logs.")

                from io import StringIO
                delimiter = ',' if ai_response.splitlines()[0].count(',') >= 1 else ';'
                csv_buffer = StringIO(ai_response)
                try:
                    df_response = pd.read_csv(csv_buffer, delimiter=delimiter)
                    df_response = df_response.drop_duplicates().dropna()
                    df_response.columns = ['Original', 'Consolidated']
                    batch_results.append(df_response)
                    # Save each individual batch
                    batch_csv_name = f'batch_{batch_num:03d}_{file}'
                    df_response.to_csv(batch_csv_name, index=False, encoding='utf-8')
                    print(f"   -> Batch {batch_num} processed and saved as '{batch_csv_name}'. [{time.time() - start_time:.1f} sec]")
                except Exception as e_csv:
                    print(f"   ❌ Error processing CSV from batch {batch_num}: {e_csv}")
                    print("   -> Gemini response was:")
                    print(ai_response)
                    failed_batches.append(batch_num)
                    continue  # Continue with the next batch
            except Exception as e_batch:
                print(f"   ❌ Error processing batch {batch_num}: {e_batch}")
                failed_batches.append(batch_num)
                continue

        # Concatenate all batches
        if batch_results:
            final_df = pd.concat(batch_results, ignore_index=True)
            final_df = final_df.drop_duplicates().dropna()
            output_file_name = f'unique_gemini_{file}'
            final_df.to_csv(output_file_name, index=False, encoding='utf-8')
            print(f"   -> Final consolidated file saved as '{output_file_name}'")
        else:
            print("   ⚠️  No valid result was obtained from the batches.")

        # Inform about failed batches
        if failed_batches:
            print(f"\n⚠️  Failed or erroneous batches: {failed_batches}")

    except Exception as e:
        print(f"   ❌ Error processing {file}: {e}")
        if "exceeds the maximum token limit" in str(e).lower():
            print("   ⚠️  The content probably exceeds Gemini model's token limit.")
            print("       Consider splitting the file or reducing the number of lines.")

print("\n🎉 Process completed for all files.")
