import socket
import pyautogui

HOST = ''  # Listen on all IPv4 interfaces
PORT = 8000

print(f"UDP server listening on port {PORT}...")

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind((HOST, PORT))

while True:
    try:
        data, addr = udp_socket.recvfrom(1024)
        print("Received data from", addr)
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
            print("No data received from", addr)
    except Exception as e:
        print("Error:", e)
