# Plugin: plugin_storage_googledrive

Google Drive CRUD tool. Gives the LLM access to files in an authorized Drive folder.

## Tool provided

`google_drive(operation, file_id, file_name, content, folder_id) → str`

Operations: `list`, `create`, `read`, `append`, `delete`

## Gate behavior

Separate read and write gates:

```
!autogate drive read true     auto-allow list/read operations
!autogate drive write true    auto-allow create/append/delete operations
!autogate drive               show current drive gate status
```

## Dependencies

```bash
pip install google-auth google-auth-oauthlib google-api-python-client
```

## Environment variables

```
FOLDER_ID=<Google Drive folder ID>
```

## Configuration files

- `credentials.json` — OAuth2 credentials (download from Google Cloud Console)
- `token.json` — auto-generated after first auth (do not commit)

## First-time setup

1. Create a project in Google Cloud Console
2. Enable the Google Drive API
3. Create OAuth2 credentials (Desktop app)
4. Download as `credentials.json` into the mymcp directory
5. Run the server — it will open a browser for authorization on first use
6. `token.json` is created automatically

## Enable

```bash
python agentctl.py enable plugin_storage_googledrive
```
