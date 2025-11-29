import os
import requests
from internal.env_parser import parse_env_file

def read_file_content(filepath):
    """Reads the content of a given file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"Error reading {filepath}: {e}"

def get_openai_next_path_recommendation():
    """
    Connects to OpenAI GPT-5.1 to get a recommendation for the next path
    based on project context files.
    """
    env_vars = parse_env_file(file_path="internal/.env")
    openai_api_key = env_vars.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in internal/.env file.")
        return

    context_files_content = []

    # Read all .md and .py files from the docs folder
    docs_dir = "docs"
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith((".md", ".py")):
                filepath = os.path.join(root, file)
                content = read_file_content(filepath)
                if not content.startswith("File not found") and not content.startswith("Error reading"):
                    context_files_content.append(f"--- {filepath} ---\n{content}\n")

    # Read specific additional files
    specific_files = ["internal/STATUS.md", "src/beta.py"]
    for filepath in specific_files:
        content = read_file_content(filepath)
        if not content.startswith("File not found") and not content.startswith("Error reading"):
            context_files_content.append(f"--- {filepath} ---\n{content}\n")

    output_filepath = "GPT_NEXT.md"

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    prompt_content = (
        "Based on the following project context files, please recommend the next logical development path or project. "
        "Be concise and provide a clear, actionable recommendation. Focus on areas that would most effectively advance the 'music manifold' concept.\n\n"
        + "\n".join(context_files_content)
    )

    data = {
        "model": "gpt-5.1",
        "messages": [
            {"role": "system", "content": "You are an expert software architect and project manager, specializing in music technology, machine learning, and mathematical modeling."},
            {"role": "user", "content": prompt_content}
        ],
        "max_completion_tokens": 100000, # User specified 100000 tokens
        "temperature": 0.7,
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_data = response.json()
        recommendation = response_data['choices'][0]['message']['content'].strip()

        with open(output_filepath, 'w') as f:
            f.write("# Recommended Next Path (from GPT-5.1)\n\n") # Updated model name in comment
            f.write(recommendation)
        print(f"Recommendation written to {output_filepath}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenAI API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
    except KeyError as e:
        print(f"Error parsing OpenAI API response: Missing key {e}. Response: {response_data}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    get_openai_next_path_recommendation()
