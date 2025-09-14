import os


def create_test_structure():
    # Tạo thư mục tests và subdirectories
    dirs = [
        'tests',
        'tests/results',
        'tests/test_images',
        'tests/sample_outputs'
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created: {d}")

    # Tạo __init__.py cho tests
    with open('tests/__init__.py', 'w') as f:
        f.write('# Test package\\n')

    print("Test structure created successfully!")


if __name__ == "__main__":
    create_test_structure()