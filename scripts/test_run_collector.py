import subprocess
import pandas as pd
import os
import time

def run_realtime_predict():
    print("\n▶️ Running realtime_predict.py...")
    subprocess.run(["python", "realtime_predict.py"])

def load_temp_result():
    temp_path = '../collected/temp_run_result.csv'
    if os.path.exists(temp_path):
        df = pd.read_csv(temp_path)
        return df.iloc[0].to_dict()
    else:
        print("❌ Temp result not found! Make sure realtime_predict.py saves temp_run_result.csv.")
        return None

def main():
    all_results = []

    while True:
        # 1. Run realtime_predict.py
        run_realtime_predict()
        time.sleep(2)  # Small delay to ensure temp file is written

        # 2. Load the result
        result = load_temp_result()
        if result is None:
            continue

        # 3. Ask feedback
        real_or_fake = input("\n🧠 Was the presented face REAL (1) or FAKE (0)?: ").strip()
        if real_or_fake not in ['0', '1']:
            print("Invalid input, skipping this run...")
            continue

        result['real_label'] = int(real_or_fake)
        all_results.append(result)

        # 4. Ask if user wants to run another test
        more = input("\n🔄 Do you want to run another test? (y/n): ").strip().lower()
        if more != 'y':
            break

    # 5. Save all results collected
    if all_results:
        save_path = '../collected/collected_results.csv'
        os.makedirs('../collected', exist_ok=True)

        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path)
            new_df = pd.DataFrame(all_results)
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = pd.DataFrame(all_results)

        final_df.to_csv(save_path, index=False)
        print(f"\n✅ All results saved to: {save_path}")

if __name__ == "__main__":
    main()
