import serial.tools.list_ports
import elliptec

def main():
    ports = [p.device for p in serial.tools.list_ports.comports()]
    print("Available serial ports:", ports)
    
    found_any = False
    for port in ports:
        print(f"\nScanning port {port} for Elliptec devices...")
        try:
            controller = elliptec.Controller(port)
        except Exception as e:
            print(f"  Could not open controller on {port}: {e}")
            continue
            
        for addr in '0123456789ABCDEF':
            try:
                rotator = elliptec.Rotator(controller, address=addr)
                angle = rotator.get_angle()
                print(f"  [SUCCESS] Found Elliptec device on {port} at Address '{addr}' with angle: {angle} degrees")
                found_any = True
            except Exception:
                pass
                
    if not found_any:
        print("\nNo Elliptec devices were detected on any COM port.")

if __name__ == '__main__':
    main()
