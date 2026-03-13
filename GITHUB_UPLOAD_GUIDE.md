# GitHub Upload Guide

This release workspace has been cleaned for GitHub publication.

## Suggested steps
1. Open this folder in a terminal:
   `tbme_submission_release_v1`
2. Initialize git if needed:
   `git init`
3. Add files:
   `git add .`
4. Commit:
   `git commit -m "TBME submission release v1"`
5. Create a GitHub repository manually on the web.
6. Add remote:
   `git remote add origin <your-repo-url>`
7. Push:
   `git branch -M main`
   `git push -u origin main`

## Note
The current environment does not provide the GitHub CLI (`gh`), so remote repository creation and push authentication must be completed with standard git or GitHub Desktop.
