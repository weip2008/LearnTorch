* merge master to main

```dos
git pull origin main
git merge --allow-unrelated-histories main
git merge main
git checkout main
git merge master
```

* Override local changes for the <file>

```dos
curl -o <local-file-path> <raw-file-url>
```

* Reset your local branch to match the remote branch:

```dos
git reset --hard origin/<branch-name>
```

* Discard all local changes and reset github:

```dos
git reset --hard HEAD
git push --force
```

