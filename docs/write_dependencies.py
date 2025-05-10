import os, re

def extract_dependencies(directory):
    dependencies = []

    # Regular expressions to match the relevant fields
    package_re = re.compile(r'set\(PACKAGE_NAME\s+([^\)]+)\)')
    repo_re = re.compile(r'set\(REPO_URL\s+"([^"]+)"\)')
    tag_re = re.compile(r'set\(REPO_TAG\s+"([^"]+)"\)')

    for filename in os.listdir(directory):
        if filename.endswith(".cmake"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

                package_match = package_re.search(content)
                repo_match = repo_re.search(content)
                tag_match = tag_re.search(content)

                if package_match and repo_match and tag_match:
                    package_name = package_match.group(1).strip()
                    repo_url = repo_match.group(1).strip()
                    repo_tag = tag_match.group(1).strip()

                    dependencies.append((package_name, repo_url, repo_tag))

    return dependencies

def write_markdown_file(dependencies, output_file="dependencies.md"):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("| Package Name | Repository URL | Version |\n")
        file.write("|-------------|---------------|---------|\n")
        for package_name, repo_url, repo_tag in dependencies:
            file.write(f"| {package_name} | [{repo_url}]({repo_url}) | {repo_tag} |\n")

if __name__ == "__main__":
    cmake_directory = "../cmake"
    dependencies = extract_dependencies(cmake_directory)
    write_markdown_file(dependencies)
    print(f"Dependencies written to dependencies.md")
