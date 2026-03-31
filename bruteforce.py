import time
import itertools
import string
import random

def generate_random_password():
    # Define character sets
    letters = string.ascii_letters  # a-z, A-Z
    digits = string.digits          # 0-9
    
    # Randomly decide number of letters (0 to 8), rest will be digits
    num_letters = random.randint(0, 8)
    num_digits = 8 - num_letters
    
    # Generate the characters
    password_chars = (
        [random.choice(letters) for _ in range(num_letters)] +
        [random.choice(digits) for _ in range(num_digits)]
    )
    
    # Shuffle to ensure random order
    random.shuffle(password_chars)
    
    # Join into a string
    return ''.join(password_chars)

def brute_force_password():
    # Character set for brute-forcing
    chars = string.ascii_letters + string.digits
    
    # Generate the random password
    password = generate_random_password()
    print(f"Generated password (for testing): {password}")
    print("Starting brute-force attack...")
    
    # Start timing
    start_time = time.perf_counter()
    attempts = 0
    
    # Try all possible combinations
    for attempt in itertools.product(chars, repeat=8):
        attempts += 1
        attempt_str = ''.join(attempt)
        
        # Show progress every 1 million attempts
        if attempts % 1000000 == 0:
            print(f"Attempts: {attempts}, current: {attempt_str}")
        
        # Check if we found the password
        if attempt_str == password:
            end_time = time.perf_counter()
            time_taken = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"Password found: {attempt_str}")
            print(f"Attempts: {attempts}")
            print(f"Time taken: {time_taken:.2f} ms")
            return
    
    # This line should never be reached
    print("Password not found.")

if __name__ == "__main__":
    brute_force_password()