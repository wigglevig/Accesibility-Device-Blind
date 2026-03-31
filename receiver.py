import socket
import pyautogui

HOST = ''  # Listen on all interfaces
PORT = 8000

print(f"Mac server listening on port {PORT}...")

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

while True:
    try:
        conn, addr = server_socket.accept()
        print("Connected by", addr)
        try:
            # Read a single command from the connection.
            data = conn.recv(1024)
            if data:
                command = data.decode().strip()
                print("Received command:", command)
                if command == "v":
                    pyautogui.press("v")
                    print("Simulated 'v' key press")
                elif command == "d":
                    pyautogui.press("d")
                    print("Simulated 'd' key press")
                elif command == "c_start":
                    pyautogui.keyDown("c")
                    print("Simulated 'c' key down")
                elif command == "c_end":
                    pyautogui.keyUp("c")
                    print("Simulated 'c' key up")
                else:
                    print("Unknown command:", command)
            else:
                print("No data received; closing connection.")
        except Exception as e:
            print("Error processing connection:", e)
        finally:
            conn.close()
    except Exception as e:
        print("Error accepting connection:", e)
