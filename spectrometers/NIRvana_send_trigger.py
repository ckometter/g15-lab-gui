import socket
import json
import pandas as pd
import numpy as np
from io import StringIO



DEVICE_IP = "10.236.10.143"  # this is the Nirvana computer --- might need to update if device PC reboots
PORT = 65432


def trigger(params):
    exposure_time = params["exposure_time"]
    message = json.dumps(params)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(10*60)
        #print("Setting socket timeout to", 10*60)
        s.connect((DEVICE_IP, PORT))
        s.sendall(message.encode('utf-8'))
        #print(f"[Main PC] Sent: {message}")

        # Receive CSV data
        def parse_wavelength(x):
            try:
                return float(x)
            except:
                return np.nan
                
                
                
        chunks = []
        try:
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        except socket.timeout:
            print("[Main PC] Socket timed out while receiving data.")
        except Exception as e:
            print(f"[Main PC] Error receiving data: {e}")

        data_str = b''.join(chunks).decode('utf-8')

        # print(f"[Main PC] Received raw data length: {len(data_str)}")
        # print(f"[Main PC] Received raw data preview:\n{data_str[:300]}")

        if not data_str.strip():
            raise ValueError("Received empty string from device — CSV is missing or transmission failed.")
            
        # Load into DataFrame
        # reader = pd.read_csv(StringIO(data_str)), chunksize=1000)
        # df = pd.concat(reader)
        
        ## Original version:
        #df = pd.read_csv(StringIO(data_str))
        
        ## Olivier test 1:
        # try:
            # df = pd.read_csv(StringIO(data_str))
        # except pd.errors.ParserError as e:
            # print("Error", data_str)
            # raise
            
        ## Olivier test 2:
        df = pd.read_csv(StringIO(data_str),engine='python')#,error_bad_lines=False)


        try:
            # Ensure proper types
            df['Frame'] = df['Frame'].astype(int) - 1  # 0-indexed
            df['Wavelength'] = df['Wavelength'].apply(parse_wavelength)
            df['Intensity'] = df['Intensity'].astype(np.int32)

            # Get array dimensions
            num_frames = df['Frame'].max() + 1
            num_wavelengths = df['Wavelength'].nunique()

            # Sort to ensure consistent wavelength order
            df = df.sort_values(['Frame', 'Wavelength'])

            # Reshape into (num_frames, num_wavelengths)
            intensity_array = df['Intensity'].to_numpy().reshape((num_frames, num_wavelengths))

            # Get unique sorted wavelength array
            wavelength_array = df[df['Frame'] == 0]['Wavelength'].to_numpy()

           
            return wavelength_array, intensity_array
        
        except Exception as e:
            print(f"[Main PC] Error parsing received data: {e}")
            return