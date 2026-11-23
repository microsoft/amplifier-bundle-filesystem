
## Committed edit events

After an edit, `apply-patch:applied` retains the requested `path` and `operation`.
An event for a committed write also carries the resolved `written_path`. A rename
identifies the destination. If writing that destination succeeds but removing
the original fails, a write event for the destination precedes the failed tool
result. Deletes have no written path. Earlier committed operations in a
multi-file patch keep their events even if a later operation fails.
