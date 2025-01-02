import pandas as pd

def data_loader(path):
    data_df = pd.read_csv(path)
    grouped_df = data_df.groupby(["Year", "Month", "Day"])
    filtered_data = []
    for _, daily_data in grouped_df:
        daily_irradiation = daily_data['Clearsky DNI'].tolist()
        first_non_zero_idx = None
        last_non_zero_idx = None
        for idx, irradiation in enumerate(daily_irradiation):
            if irradiation > 0:
                first_non_zero_idx = idx
                break
        for idx in range(len(daily_irradiation) - 1, -1, -1):
            if daily_irradiation[idx] > 0:
                last_non_zero_idx = idx
                break
        if first_non_zero_idx is not None and last_non_zero_idx is not None:
            non_zero_daily_irradiation = daily_irradiation[first_non_zero_idx:last_non_zero_idx + 1]
            filtered_data.append(non_zero_daily_irradiation)

        # filtered_rows = daily_data[(daily_data['Hour'] >= 3) & (daily_data['Hour'] <= 18)]


        # if not filtered_rows.empty:
        #     daily_irradiation = filtered_rows['Clearsky DNI'].tolist()
        #     non_zero_daily_irradiaiton = [irradiation for irradiation in daily_irradiation if irradiation > 0]
        #     filtered_data.append(filtered_rows['Clearsky DNI'].tolist())
        
    return filtered_data

def increase_data_resolution(data):
    fine_data = []
    for daily_data in data:
        fine_data_daily = []
        for idx in range(len(daily_data) - 1):
            current_data = daily_data[idx]
            next_data = daily_data[idx + 1]
            for inner_idx in range(15):
                fine_data_daily.append(current_data + (next_data - current_data) / 15 * (inner_idx))
        fine_data_daily.append(next_data)
        fine_data.append(fine_data_daily)
    return fine_data
            
