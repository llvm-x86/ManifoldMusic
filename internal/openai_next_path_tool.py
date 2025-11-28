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
    env_vars = parse_env_file()
    openai_api_key = env_vars.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in .env file.")
        return

    # Read context files
    temporal_content = read_file_content("docs/temporal.md")
    zyra_photon_content = read_file_content("docs/zyra_photon.md")
    beta_content = read_file_content("docs/beta.py")

    output_filepath = "docs/riemann_Information_Extraction_applications.md"
    with open(output_filepath, 'w') as f:
        f.write("# Recommended Next Path (from GPT-5.1)\n\n")
        f.write(recommendation)
    print(f"Recommendation written to {output_filepath}")
    print("\n--- GPT-5.1 Recommendation ---\n")
    print(recommendation)
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    prompt_content = (
        "Based on the following project context files, please recommend the next logical development path or project. "
        "Be concise and provide a clear, actionable recommendation.\n\n"
        f"--- docs/temporal.md ---\n{temporal_content}\n\n"
        f"--- docs/zyra_photon.md ---\n{zyra_photon_content}\n\n"
        f"--- docs/beta.py ---\n{beta_content}\n"
    )

    data = {
        "model": "gpt-5.1",
        "messages": [
            {"role": "system", "content": "You are an expert software architect and project manager."},
            {"role": "user", "content": prompt_content}
        ],
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_data = response.json()
        recommendation = response_data['choices'][0]['message']['content'].strip()

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
