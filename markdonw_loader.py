
def load_markdown(dir: str) -> list:
    # Splits the content of a markdown file in dir into lines, removes empty lines
    def is_made_of(line: str, chars: set) -> bool:
        # determines if a line is made of characters in set 
        for char in line:
            if not char in chars:
                return False
        return True
     
    with open(dir, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    lines = [line for line in lines if not is_made_of(line, {' ', '\n'}) and len(line) > 5]
    return lines


if __name__ == "__main__":
    lines = load_markdown('test.md')
    for line in lines:
        print(line)

