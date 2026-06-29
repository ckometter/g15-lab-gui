import Pyro4
import time

def debug_preview_diagnostics():
    server_ip = "G15-Pylon.dhcp-int.phys.ethz.ch"
    port = 50920
    uri = f"PYRO:LightField@{server_ip}:{port}"
    print(f"Connecting to {uri}...")
    
    Pyro4.config.SERIALIZER = 'pickle'
    proxy = Pyro4.Proxy(uri)
    
    try:
        print("Starting preview on hardware...")
        proxy.start_preview()
        
        print("Polling get_latest_preview_data() for 5 seconds...")
        for i in range(15):
            time.sleep(0.3)
            try:
                data = proxy.get_latest_preview_data()
                if data is None:
                    print(f"Poll {i}: None")
                elif isinstance(data, str):
                    print(f"Poll {i}: Received String from server:\n{data}")
                else:
                    print(f"Poll {i}: Success! Type: {type(data)}, Length: {len(data)}, Min: {min(data)}, Max: {max(data)}")
            except Exception as e:
                print(f"Poll {i} Error: {e}")
                
        print("Stopping preview on hardware...")
        proxy.stop_preview()
        print("Preview stopped.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        proxy._pyroRelease()

if __name__ == '__main__':
    debug_preview_diagnostics()
    