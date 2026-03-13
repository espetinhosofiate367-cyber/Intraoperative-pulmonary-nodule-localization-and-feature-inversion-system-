# GitHub Upload Guide

This release workspace has been cleaned for GitHub publication.

## Recommended repository name
`Intraoperative-pulmonary-nodule-localization-and-feature-inversion-system`

## Suggested steps
1. Open this folder in a terminal:
   `tbme_submission_release_v1`
2. Create a new GitHub repository on the web with this exact name:
   `Intraoperative-pulmonary-nodule-localization-and-feature-inversion-system`
3. Copy the repository URL from GitHub, then run:
   `git remote add origin <your-repo-url>`
4. Push the current local release repository:
   `git push -u origin main`

## Example
If your GitHub username is `yourname`, then the remote may look like:
- HTTPS:
  `https://github.com/yourname/Intraoperative-pulmonary-nodule-localization-and-feature-inversion-system.git`
- SSH:
  `git@github.com:yourname/Intraoperative-pulmonary-nodule-localization-and-feature-inversion-system.git`

## Note
The current environment does not provide the GitHub CLI (`gh`), so remote repository creation and push authentication must be completed with standard git or GitHub Desktop.
