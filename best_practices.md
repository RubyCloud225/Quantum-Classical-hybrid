# 🧑‍💻 Solo-Dev Best Practices Checklist

## 1. Version Control
- [ ] Use Git for *everything*.
- [ ] Commit small, logical changes with meaningful messages.
- [ ] Use branches for experiments / big features.
- [ ] Delete code instead of commenting it out — Git has your history.

## 2. Code Style
- [ ] Use an auto-formatter / linter (`clang-format`, `black`, `eslint`, etc.).
- [ ] Keep function names, variables, and classes descriptive.
- [ ] Stick to one consistent brace/indent style.
- [ ] Limit functions to one responsibility.

## 3. Comments & Documentation
- [ ] Comment **why**, not **what**.
- [ ] Remove temporary debugging comments when done.
- [ ] Write a short `README.md` for each project (setup, usage, known issues).
- [ ] Add docstrings at the module/class/function level (esp. public ones).

## 4. Testing & Debugging
- [ ] Write small tests (unit/integration) where bugs could creep in.
- [ ] Automate tests if possible (local first, CI/CD later).
- [ ] Use assertions to catch unexpected states early.

## 5. Structure & Organization
- [ ] Keep related code together in modules / folders.
- [ ] Use clear file names (`data_loader.cpp`, not `dl.cpp`).
- [ ] Avoid “god classes” — break down big logic.
- [ ] Document project structure in README if non-trivial.

## 6. Work Habits
- [ ] Don’t leave long-term commented-out code (stash it in Git).
- [ ] Refactor opportunistically (when it helps, not endlessly).
- [ ] Log progress / design choices in a `NOTES.md` so you remember why.
- [ ] Automate repetitive tasks with scripts (build, run tests, deploy).
