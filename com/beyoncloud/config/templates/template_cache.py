import os
import glob
import aiofiles
import json
import logging
import httpx

logger = logging.getLogger(__name__)

class TemplateCache:
    def __init__(self, base_path: str, preload: list[str] = None, load_all: str = "N"):
        """
        :param base_path: Base config folder containing all template subfolders
        :param preload: List of template names to preload (ignored if load_all=True)
        :param load_all: If True, load all templates at startup
        """
        self.base_path = base_path
        self.load_all = load_all
        self.preload = [t.lower() for t in preload] if preload else []
        self.templates = {}

    async def _decrypt_file_if_needed(self, folder_path:str ,enc_file: str) -> str:
        """
        Call Auth Service DECRYPT_TEMPLATE if encrypted file found.
        Returns path to decrypted .ndjson file.
        """

        payload = {
            "session_id": "cache_loader",
            "org_id": 1,
            "user_id": 1,
            "user_name": "system",
            "template_name": "DECRYPT_TEMPLATE",
            "input_file_path": folder_path,
            "input_file_name": enc_file,
            "output_file_path": "",
            "output_file_name": "",
            "input_text": ""
        }

        async with httpx.AsyncClient() as client:
            r = await client.post("http://127.0.0.1:5015/api/v1/auth/auth-service", json=payload)
            if r.status_code != 200:
                raise ValueError(f"Decryption failed for {enc_file}: {r.text}")
            res_json = r.json()
            return os.path.join(res_json["output_file_path"], res_json["output_file_name"])

    async def _read_ndjson_file(self, file_path: str) -> list[dict]:
        """Read an NDJSON file asynchronously and return parsed JSON records."""
        records = []
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
        return records

    async def _load_encrypted_file(self, folder_path: str, enc_file: str) -> list[dict]:
        """Decrypt and load an encrypted NDJSON file, then clean up."""
        try:
            logger.info(f"Decrypting encrypted template file: {enc_file}")
            ndjson_path = await self._decrypt_file_if_needed(folder_path, enc_file)
            records = await self._read_ndjson_file(ndjson_path)
        except Exception as e:
            logger.error(f"Failed to decrypt {enc_file}: {e}")
            return []
        finally:
            if 'ndjson_path' in locals() and os.path.exists(ndjson_path):
                os.remove(ndjson_path)
                logger.info(f"Temporary decrypted file deleted: {ndjson_path}")
        return records

    async def _load_template_from_disk(self, template_folder: str) -> list[dict]:
        """
        Read NDJSON or encrypted files (.ndjson.hybrid.enc) for a given template folder.
        """
        folder_path = os.path.join(self.base_path, template_folder)
        if not os.path.isdir(folder_path):
            logger.warning(f"Template folder {template_folder} not found in {self.base_path}")
            return []

        records = []

        # Load plain NDJSON
        plain_files = [
            f for f in glob.glob(os.path.join(folder_path, "*.ndjson"))
            if not f.endswith(".ndjson.hybrid.enc")
        ]
        for ndjson_file in plain_files:
            records.extend(await self._read_ndjson_file(ndjson_file))

        # Load encrypted NDJSON
        enc_files = glob.glob(os.path.join(folder_path, "*.ndjson.hybrid.enc"))
        for enc_file in enc_files:
            records.extend(await self._load_encrypted_file(folder_path, enc_file))

        return records

    async def load_templates(self):
        """Preload selected templates or all templates at startup."""
        logger.info(f"------------------------> Loading templates from {self.load_all}...")
        if self.load_all=="Y":
            template_folders =self.list_templates()
        else:
            template_folders = self.preload

        for template_name in template_folders:
            logger.info(f"Preloading template: {template_name}")
            self.templates[template_name.lower()] =await self._load_template_from_disk(template_name)
            logger.info(f"Loaded {len(self.templates[template_name.lower()])} records for {template_name}")

    async def get_template(self, template_name: str):
        """Return dataset for a template (preloaded or lazy-loaded)."""
        key = template_name.lower()
        if key not in self.templates:
            logger.info(f"Lazy loading template: {key}")
            self.templates[key] =await self._load_template_from_disk(template_name)
        return self.templates[key]

    def list_templates(self):
        """List all available template directories in base_path."""
        return [
            f for f in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, f))
        ]

    async def refresh_template(self, template_name: str = None):
        """
        Force reload.
        - If template_name is given → reload only that template
        - If template_name=None → reload all templates
        - This should be called only when config server detects changes
        """
        if template_name:
            key = template_name.lower()
            logger.info(f"Refreshing template: {key}")
            self.templates[key] =await self._load_template_from_disk(template_name)
            return self.templates[key]
        else:
            logger.info("Refreshing ALL templates")
            self.templates.clear()
            self.load_templates()
            return self.templates
