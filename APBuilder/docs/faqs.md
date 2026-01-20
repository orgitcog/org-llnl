# Frequently Asked Questions (FAQ)

## What is Volume Mapping in Containers?

Volume mapping allows you to link a directory or file from your host machine (the `source`) to a
directory inside a container (the `target`). This makes it possible for the container to read
and write files that are stored on your local filesystem. Any changes made by the container in
the mapped (`target`) directory are immediately reflected on your host (`source`), and vice versa.
This is useful for sharing data, saving output, or persisting files even after the container stops.
Here is example volume mapping:

```yaml
services:
  apbuilder:
    volumes:
      - type: bind
        source: /tmp/apbuilder/data
        target: /home/apbuilder/apbuilder/data
```

## The data is not saved to the specified output directory when using the container

When using the container image, you must specify in `volume-overrides.yaml` the `source` (a path
on your local machine) that will be mapped to the `target` (the path inside the container).
The output data is saved to the `target` directory as seen by the container. However, because
of the volume mapping, any data written to the `target` directory inside the container is actually
stored in your `source` directory on your local filesystem.  

Therefore, the `-out` directory must be the absolute path **inside the container** (the `target`
path), not the host (`source`) path. For example, consider the following `volume-overrides.yaml`:

```yaml
services:
  apbuilder:
    volumes:
      - type: bind
        source: /tmp/apbuilder/data
        target: /home/apbuilder/apbuilder/data
```

If you want the output data to be accessible on your local machine at
`/tmp/apbuilder/data/myanalysis`, you should set the `-out` parameter to
`/home/apbuilder/apbuilder/data/myanalysis` (the `target` path inside the container).
The container will write to this directory, and the data will appear in
`/tmp/apbuilder/data/myanalysis` on your host machine.

---

### Table Summary

| Parameter         | Value inside container           | Value on host machine         |
|-------------------|---------------------------------|------------------------------|
| `-out` argument   | `/home/apbuilder/apbuilder/data/myanalysis` | `/tmp/apbuilder/data/myanalysis` |
| Data physically stored | `/home/apbuilder/apbuilder/data/myanalysis` (container) | `/tmp/apbuilder/data/myanalysis` (host) |

---

**In summary:**  
Always use the `target` (container path) for the `-out` parameter. The data will be available on
your host in the `source` directory you mapped.

## Using Windows Git Bash is modifying my output directory when using the container

The specified `-out` directory can be automatically modified when using Git Bash for Windows to
run the container. This is because Git Bash tries to "translate" Unix-style paths (like /home/...)
to Windows-style paths (C:/Program Files/Git/home/...). When you run a command that includes a
path starting with `/`, Git Bash assumes you mean a Windows absolute path, and prepends the Git
installation directory if it can't resolve the path. Using `//` at the start (double slash) can
prevent Git Bash from translating the path:

```bash
-out //home/apbuilder/apbuilder/data/raul-data
```

## Getting SSL errors when downloading data

If you are connected to VPN and having SSL errors, you will have to configure an environment
variable with the SSL certificate.

Here is an example to configure on Linux. Use the defined approach for your OS to set the
environment variable accordingly.

```bash
export REQUESTS_CA_BUNDLE=/path/to/cacert.pem
```
