"""Deliverers: `email` (SMTP, markdown->HTML) and `podcast_feed` (GitHub release
asset upload + RSS 2.0 feed maintenance). See CONTRACTS.md for signatures.

`email` lets SMTP errors propagate (a failed send must not be recorded as
delivered). `podcast_feed` degrades silently — a missing release/feed shouldn't
fail the run — logging any failure once here.
"""

import logging
import os
import smtplib
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate

import markdown

from .models import Result
from .registry import deliverers

logger = logging.getLogger(__name__)

_RSS_SKELETON = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
<channel>
<title>{title}</title>
<link>{link}</link>
<description>{description}</description>
<itunes:author>{title}</itunes:author>
</channel>
</rss>
"""


# == Email ====================================================================


@deliverers.register("email")
def email(result: Result, cfg: dict) -> None:
    """Send result.markdown (rendered to HTML) as a multipart/alternative email."""
    conf = cfg["delivery"]["email"]
    if not result.markdown or not result.markdown.strip():
        logger.info("email: empty markdown, nothing to send")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = result.subject or "Knowledge Secretary"
    msg["From"] = conf["from"]
    msg["To"] = ", ".join(conf["to"])
    # plain first, html second: clients render the last part they understand
    msg.attach(MIMEText(result.markdown, "plain"))
    msg.attach(MIMEText(markdown.markdown(result.markdown, extensions=["extra"]), "html"))

    with smtplib.SMTP(conf["smtp_host"], conf["smtp_port"]) as smtp:
        smtp.starttls()
        if conf.get("username") and conf.get("password"):
            smtp.login(conf["username"], conf["password"])
        smtp.sendmail(conf["from"], conf["to"], msg.as_string())
    logger.info("✅ email: sent %r to %s", msg["Subject"], conf["to"])


# == Podcast feed =============================================================


@deliverers.register("podcast_feed")
def podcast_feed(result: Result, cfg: dict) -> None:
    """Upload artifacts[0] as a GitHub release asset, then prepend an RSS <item>."""
    conf = cfg["delivery"]["podcast_feed"]
    if not result.artifacts:
        logger.info("podcast_feed: no artifacts, skipping")
        return

    mp3_path = result.artifacts[0]
    if conf.get("site_dir"):
        os.makedirs(conf["site_dir"], exist_ok=True)

    asset_url = _upload_release_asset(mp3_path, result, conf)
    if asset_url is None:
        return  # failure already logged by _upload_release_asset
    _prepend_rss_item(asset_url, mp3_path, result, conf)


# == Helper Functions =========================================================

# ----- podcast_feed -----


def _upload_release_asset(mp3_path: str, result: Result, conf: dict) -> str | None:
    """Create (or update) a dated GH release with mp3_path attached; return the
    asset's public download URL, or None on failure."""
    repo = conf.get("github_repo") or os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        logger.warning("⚠️ podcast_feed: no github_repo configured, skipping upload")
        return None

    tag = "podcast-" + datetime.now(UTC).strftime("%Y-%m-%d")
    title = result.subject or result.meta.get("topic") or tag
    notes = result.meta.get("topic") or title
    try:
        create = subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                mp3_path,
                "--repo",
                repo,
                "--title",
                title,
                "--notes",
                notes,
            ],
            capture_output=True,
            text=True,
        )
        if create.returncode != 0:
            # most likely today's tag already exists (same-day rerun) -> replace asset.
            # NOTE: also catches genuine auth/repo errors, which then fail the upload below.
            upload = subprocess.run(
                ["gh", "release", "upload", tag, mp3_path, "--repo", repo, "--clobber"],
                capture_output=True,
                text=True,
            )
            if upload.returncode != 0:
                logger.warning(
                    "⚠️ podcast_feed: gh release create+upload failed: %s / %s",
                    create.stderr.strip(),
                    upload.stderr.strip(),
                )
                return None
    except Exception as e:
        logger.warning("⚠️ podcast_feed: gh release error: %s", e)
        return None

    return f"https://github.com/{repo}/releases/download/{tag}/{os.path.basename(mp3_path)}"


def _prepend_rss_item(asset_url: str, mp3_path: str, result: Result, conf: dict) -> None:
    """Parse (or create) the RSS feed at conf['feed_path'] and prepend a new <item>."""
    feed_path = conf["feed_path"]
    try:
        if os.path.dirname(feed_path):
            os.makedirs(os.path.dirname(feed_path), exist_ok=True)
        if os.path.exists(feed_path):
            root = ET.parse(feed_path).getroot()
        else:
            root = ET.fromstring(
                _RSS_SKELETON.format(
                    title=conf.get("podcast_title", "Podcast"),
                    link=conf.get("base_url", ""),
                    description=conf.get("podcast_title", "Podcast"),
                )
            )
        channel = root.find("channel")
        if channel is None:
            logger.warning("⚠️ podcast_feed: malformed feed (no <channel>), skipping")
            return

        try:
            length = str(os.path.getsize(mp3_path))
        except OSError:
            length = "0"
        title = result.meta.get("topic") or result.subject or "Episode"
        item = ET.Element("item")
        ET.SubElement(item, "title").text = title
        ET.SubElement(item, "description").text = title
        ET.SubElement(item, "guid").text = asset_url
        ET.SubElement(item, "enclosure", {"url": asset_url, "type": "audio/mpeg", "length": length})
        ET.SubElement(item, "pubDate").text = formatdate(datetime.now(UTC).timestamp())

        insert_at = next((i for i, c in enumerate(channel) if c.tag == "item"), len(channel))
        channel.insert(insert_at, item)
        ET.ElementTree(root).write(feed_path, encoding="UTF-8", xml_declaration=True)
        logger.info("✅ podcast_feed: published %r", title)
    except Exception as e:
        logger.warning("⚠️ podcast_feed: RSS update failed: %s", e)
