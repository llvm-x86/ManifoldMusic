
def parse_env_file(file_path=".env"):
    """
    Parses a .env file and returns a dictionary of key-value pairs.
    Handles basic KEY=VALUE format, comments, and ignores empty lines.
    """
    env_vars = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Warning: .env file not found at {file_path}")
    return env_vars

# Example usage (for testing purposes, remove in final tool script)
# if __name__ == "__main__":
#     # Create a dummy .env file for testing
#     with open(".env", "w") as f:
#         f.write("OPENAI_API_KEY=sk-testkey123\n")
#         f.write("# This is a comment\n")
#         f.write("ANOTHER_VAR = some_value\n")
#         f.write("\n")
#         f.write("EMPTY_VALUE=\n")

#     env = parse_env_file()
#     print(env)
#     # Expected: {'OPENAI_API_KEY': 'sk-testkey123', 'ANOTHER_VAR': 'some_value', 'EMPTY_VALUE': ''}

#     import os
#     os.remove(".env") # Clean up dummy file
