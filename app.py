import itertools
import math
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "for",
    "from",
    "get",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "make",
    "new",
    "not",
    "of",
    "on",
    "or",
    "our",
    "out",
    "over",
    "the",
    "this",
    "to",
    "today",
    "top",
    "up",
    "what",
    "when",
    "why",
    "with",
    "you",
    "your",
}

TOPIC_TEMPLATES = [
    "Beginner's guide to {kw1}",
    "{kw1} vs {kw2}: which should you choose?",
    "How to master {kw1} in 30 days",
    "The truth about {kw1} and {kw2}",
    "{kw1} mistakes creators keep making",
    "{kw1} trends to watch this year",
    "Step-by-step {kw1} workflow",
    "{kw1} tools that save hours",
]


def build_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def get_uploads_playlist_id(client, channel_id: str) -> str:
    response = (
        client.channels()
        .list(part="contentDetails", id=channel_id)
        .execute()
    )
    items = response.get("items", [])
    if not items:
        raise ValueError("No channel found for the provided channel ID.")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def extract_video_id(video_url: str) -> str:
    match = re.search(r"(?:v=|/)([a-zA-Z0-9_-]{11})", video_url)
    if not match:
        raise ValueError("Unable to extract a video ID from the provided URL.")
    return match.group(1)


def get_channel_id_from_video(client, video_url: str) -> str:
    video_id = extract_video_id(video_url)
    response = client.videos().list(part="snippet", id=video_id).execute()
    items = response.get("items", [])
    if not items:
        raise ValueError("No video found for the provided URL.")
    return items[0]["snippet"]["channelId"]


def fetch_playlist_video_ids(client, playlist_id: str, max_videos: int = 50) -> List[str]:
    video_ids: List[str] = []
    next_page = None
    while len(video_ids) < max_videos:
        response = (
            client.playlistItems()
            .list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=min(50, max_videos - len(video_ids)),
                pageToken=next_page,
            )
            .execute()
        )
        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
        next_page = response.get("nextPageToken")
        if not next_page:
            break
    return video_ids


def fetch_video_details(client, video_ids: List[str]) -> List[Dict]:
    details: List[Dict] = []
    for chunk_start in range(0, len(video_ids), 50):
        chunk_ids = video_ids[chunk_start : chunk_start + 50]
        response = (
            client.videos()
            .list(part="snippet,statistics", id=",".join(chunk_ids))
            .execute()
        )
        details.extend(response.get("items", []))
    return details


def build_dataframe(details: List[Dict]) -> pd.DataFrame:
    rows = []
    for item in details:
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        rows.append(
            {
                "video_id": item.get("id"),
                "title": snippet.get("title"),
                "published_at": snippet.get("publishedAt"),
                "tags": ", ".join(snippet.get("tags", [])),
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        df.sort_values("published_at", ascending=False, inplace=True)
    return df


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "avg_views": 0,
            "avg_likes": 0,
            "avg_comments": 0,
            "like_comment_ratio": 0,
        }
    avg_views = df["views"].mean()
    avg_likes = df["likes"].mean()
    avg_comments = df["comments"].mean()
    like_comment_ratio = avg_likes / avg_comments if avg_comments else math.nan
    return {
        "avg_views": avg_views,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "like_comment_ratio": like_comment_ratio,
    }


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def weighted_keyword_scores(df: pd.DataFrame) -> Counter:
    now = pd.Timestamp.utcnow()
    scores: Counter = Counter()
    for _, row in df.iterrows():
        published = row["published_at"]
        months_since = (now - published).days / 30 if pd.notnull(published) else 0
        weight = 1 / (1 + months_since)
        tokens = tokenize(str(row["title"]))
        if row["tags"]:
            tokens.extend(tokenize(row["tags"]))
        for token in tokens:
            scores[token] += weight
    return scores


def build_topic_predictions(df: pd.DataFrame, max_topics: int = 8) -> Tuple[List[str], List[Tuple[str, float]]]:
    if df.empty:
        return [], []
    scores = weighted_keyword_scores(df)
    top_keywords = scores.most_common(8)
    keywords = [kw for kw, _ in top_keywords]
    topics = []
    for template, pair in zip(TOPIC_TEMPLATES, itertools.cycle(itertools.combinations(keywords, 2))):
        kw1, kw2 = pair
        topic = template.format(kw1=kw1.title(), kw2=kw2.title())
        if topic not in topics:
            topics.append(topic)
        if len(topics) >= max_topics:
            break
    return topics, top_keywords


def plot_upload_frequency(df: pd.DataFrame):
    if df.empty:
        st.info("No videos to plot.")
        return
    df_monthly = (
        df.set_index("published_at")
        .resample("M")
        .size()
        .rename("videos")
        .reset_index()
    )
    st.line_chart(df_monthly, x="published_at", y="videos")


def plot_views_over_time(df: pd.DataFrame):
    if df.empty:
        st.info("No videos to plot.")
        return
    st.line_chart(df, x="published_at", y="views")


def main():
    st.set_page_config(page_title="YouTube Insight Generator", layout="wide")
    st.title("YouTube Video Analysis & Insight Generator")
    st.write(
        "Analyze channel performance, engagement trends, and get next video topic predictions."
    )

    with st.sidebar:
        st.header("Inputs")
        api_key = st.text_input("YouTube Data API Key", type="password")
        video_url = st.text_input("Video URL")
        max_videos = st.slider("Number of recent videos", min_value=10, max_value=100, value=50)
        fetch_button = st.button("Fetch Insights")

    if fetch_button:
        if not api_key or not video_url:
            st.error("Please provide both an API key and a video URL.")
            return

        with st.spinner("Fetching data from YouTube..."):
            try:
                client = build_client(api_key)
                channel_id = get_channel_id_from_video(client, video_url)
                uploads_playlist = get_uploads_playlist_id(client, channel_id)
                video_ids = fetch_playlist_video_ids(client, uploads_playlist, max_videos)
                details = fetch_video_details(client, video_ids)
                df = build_dataframe(details)
            except Exception as exc:
                st.error(f"Unable to fetch data: {exc}")
                return

        st.subheader("Channel Summary")
        metrics = compute_metrics(df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Views", f"{metrics['avg_views']:.0f}")
        col2.metric("Average Likes", f"{metrics['avg_likes']:.0f}")
        col3.metric("Average Comments", f"{metrics['avg_comments']:.0f}")
        ratio = metrics["like_comment_ratio"]
        ratio_display = f"{ratio:.2f}" if math.isfinite(ratio) else "N/A"
        col4.metric("Like-to-Comment Ratio", ratio_display)

        st.subheader("Upload Frequency")
        plot_upload_frequency(df)

        st.subheader("Views Over Time")
        plot_views_over_time(df)

        st.subheader("Next Video Topic Prediction")
        topics, keywords = build_topic_predictions(df)
        if topics:
            st.write("Suggested topics based on recent performance and keyword momentum:")
            st.markdown("\n".join([f"- {topic}" for topic in topics]))
            st.write("Top weighted keywords:")
            st.dataframe(pd.DataFrame(keywords, columns=["Keyword", "Score"]))
        else:
            st.info("Not enough data to generate topic predictions.")

        st.subheader("Latest Video Details")
        st.dataframe(df[["title", "published_at", "views", "likes", "comments"]])


if __name__ == "__main__":
    main()

   
