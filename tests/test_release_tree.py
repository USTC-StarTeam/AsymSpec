from pathlib import Path
import os
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".yaml", ".yml", ".txt"}


def release_files():
    for directory, subdirs, filenames in os.walk(ROOT):
        subdirs[:] = [
            name for name in subdirs
            if name not in {".git", "__pycache__", ".venv"}
        ]
        base = Path(directory)
        for filename in filenames:
            yield base / filename


class ReleaseTreeTest(unittest.TestCase):
    def test_required_release_files_exist(self):
        for name in [
            "README.md", "LICENSE", "NOTICE", "AUTHORS.md",
            "CITATION.cff", "RELEASE_PROVENANCE.md", "requirements.txt",
            "VLLM_COMPATIBILITY.md",
        ]:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_no_machine_specific_paths(self):
        forbidden = ["/root/dataDisk", "/mnt/disk0", "/mnt/src/", "PPTAgent/"]
        failures = []
        for path in release_files():
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(errors="ignore")
            for token in forbidden:
                if token in text:
                    failures.append(f"{path.relative_to(ROOT)}: {token}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_no_common_secret_shapes(self):
        patterns = [
            re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
            re.compile(r"hf_[A-Za-z0-9]{20,}"),
            re.compile(r"AKIA[0-9A-Z]{16}"),
            re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
        ]
        failures = []
        for path in release_files():
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(errors="ignore")
            if any(pattern.search(text) for pattern in patterns):
                failures.append(str(path.relative_to(ROOT)))
        self.assertEqual(failures, [])

    def test_no_large_release_artifacts(self):
        large = [
            str(path.relative_to(ROOT))
            for path in release_files()
            if path.is_file()
            and path.stat().st_size > 5 * 1024 * 1024
        ]
        self.assertEqual(large, [])


if __name__ == "__main__":
    unittest.main()
