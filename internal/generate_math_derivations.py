import os
import requests
import argparse
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

def generate_derivation(input_filepath, output_filepath, model_name="gpt-4-turbo", prompt_prefix=""):
    """
    Connects to OpenAI API to get a mathematical derivation for the given file content.
    """
    env_vars = parse_env_file(file_path="internal/.env")
    openai_api_key = env_vars.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in .env file.")
        return

    file_content = read_file_content(input_filepath)
    if file_content.startswith("File not found") or file_content.startswith("Error reading"):
        print(file_content)
        return

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    base_prompt = (
        f"Please provide a thorough mathematical derivation and explanation for the concepts "
        f"present in the following file, named '{os.path.basename(input_filepath)}'. "
        f"Focus on underlying mathematical principles, formulas, and their interrelations. "
        f"If the file contains code, derive the mathematical basis of the algorithms. "
        f"If it's a placeholder, propose relevant mathematical concepts based on the file name and project context (music manifold). "
        f"Structure the output in Markdown with appropriate headings and LaTeX for equations.\n\n"
        f"--- Content of {os.path.basename(input_filepath)} ---\n{file_content}\n"
    )

    if prompt_prefix:
        final_prompt = prompt_prefix + "\n\n" + base_prompt
    else:
        final_prompt = base_prompt

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are an expert mathematician and physicist, capable of deriving complex mathematical and physical concepts from code and descriptions."},
            {"role": "user", "content": final_prompt}
        ],
        "max_completion_tokens": 10000,
        "temperature": 0.7,
    }

    print(f"Generating derivation for {input_filepath} using model {model_name}...")
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_data = response.json()
        derivation_content = response_data['choices'][0]['message']['content'].strip()

        with open(output_filepath, 'w') as f:
            f.write(derivation_content)
        print(f"Derivation written to {output_filepath}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenAI API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
    except KeyError as e:
        print(f"Error parsing OpenAI API response: Missing key {e}. Response: {response_data}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mathematical derivation using OpenAI API.")
    parser.add_argument("input_filepath", help="Path to the input file (e.g., code, markdown).")
    parser.add_argument("output_filepath", help="Path to the output markdown file for the derivation.")
    parser.add_argument("--model", default="gpt-5.1", help="OpenAI model to use (default: gpt-5.1).")
    parser.add_argument("--prompt_prefix", default="", help="Additional text to prepend to the OpenAI prompt.")
    args = parser.parse_args()

    generate_derivation(args.input_filepath, args.output_filepath, args.model, args.prompt_prefix)
