import threading

import spotipy
import streamlit as st
import streamlit.components.v1 as components
from index import load_index
from inference import (
    generate_prompt,
    load_dataset_map,
    load_llm,
    load_retriever,
    rank_candidates,
    retrieve_candidates,
)
from peft.auto import AutoPeftModelForCausalLM
from playlists import load_playlist_map
from spotipy.oauth2 import SpotifyClientCredentials
from torch.jit import ScriptModule
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from whoosh.index import FileIndex
from whoosh.qparser import QueryParser

MAX_SEARCH_RESULTS = 20
MAX_RETRIEVER_CANDIDATES = 20
MAX_RECOMMENDATIONS = 10
MAX_PLAYLIST_SIZE = 50


def get_recommendations(songs_ids: list[int]):
    with st.session_state.lock_recommender:
        candidates = retrieve_candidates(retriever, songs_ids, MAX_RETRIEVER_CANDIDATES)
        llm_prompt = generate_prompt(songs_ids, candidates, dataset_map)
        llm_recommendations = rank_candidates(
            llm, tokenizer, llm_prompt, candidates, dataset_map, MAX_RECOMMENDATIONS
        )
        st.session_state.recommendations = llm_recommendations
        st.session_state.prompt = llm_prompt


def search_songs():
    with ix.searcher() as searcher:
        query = QueryParser("song", ix.schema).parse(st.session_state.search_input)
        results = searcher.search(query, limit=MAX_SEARCH_RESULTS)
        st.session_state.search_results = [
            (hit["id"], hit["song"]) for hit in results if hit["song"]
        ]


def load_playlist(playlist_name: str):
    st.session_state.playlist_songs = playlist_map[playlist_name][:]


def add_to_playlist(song_num: int, song_name: str):
    if len(st.session_state.playlist_songs) >= MAX_PLAYLIST_SIZE:
        st.warning("Playlist is full, remove a song to add another.")
        return
    st.session_state.playlist_songs.append((song_num, song_name))


def remove_from_playlist(index: int):
    if st.session_state.playlist_songs:
        st.session_state.playlist_songs.pop(index)


def lookup_spotify_uri(song_name: str) -> str:
    results = sp.search(q=f"track:{song_name}", limit=1)
    if results and results["tracks"]["items"]:
        return results["tracks"]["items"][0]["uri"]
    return ""


@st.cache_resource
def load_resources() -> tuple[
    FileIndex,
    dict[int, str],
    dict[str, list[tuple[int, str]]],
    ScriptModule,
    AutoPeftModelForCausalLM,
    PreTrainedTokenizer | PreTrainedTokenizerFast,
]:
    ix = load_index()
    dataset_map = load_dataset_map()
    playlist_map = load_playlist_map(dataset_map)
    retriever = load_retriever()
    llm, tokenizer = load_llm()
    return ix, dataset_map, playlist_map, retriever, llm, tokenizer


@st.cache_resource
def auth_spotify() -> spotipy.Spotify:
    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=st.secrets["SPOTIPY_CLIENT_ID"],
            client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
        )
    )
    return sp


st.set_page_config(
    layout="wide",
    page_title="LlamaRec Demo",
    menu_items={"About": "https://github.com/GarciaLnk/LlamaRec"},
)

st.session_state.search_results = st.session_state.get("search_results", [])
st.session_state.playlist_songs = st.session_state.get("playlist_songs", [])
st.session_state.recommendations = st.session_state.get("recommendations", [])
st.session_state.lock_recommender = st.session_state.get(
    "lock_recommender", threading.Lock()
)

ix, dataset_map, playlist_map, retriever, llm, tokenizer = load_resources()
sp = auth_spotify()

st.title("LlamaRec Demo")

col_res, col_list, col_recsys = st.columns([1, 1, 1], gap="large")

col_res.write("Select a pre-defined playlist...")
with col_res:
    col_rock, col_pop, col_rap, col_spanish = st.columns([1, 1, 1, 1], gap="small")
    col_rock.button(
        "Rock",
        key="rock_playlist",
        on_click=load_playlist,
        args=("Rock",),
        use_container_width=True,
    )
    col_pop.button(
        "Pop",
        key="pop_playlist",
        on_click=load_playlist,
        args=("Pop",),
        use_container_width=True,
    )
    col_rap.button(
        "Rap",
        key="rap_playlist",
        on_click=load_playlist,
        args=("Rap",),
        use_container_width=True,
    )
    col_spanish.button(
        "Spanish",
        key="spanish_playlist",
        on_click=load_playlist,
        args=("Spanish",),
        use_container_width=True,
    )

col_res.text_input(
    "...or search for songs to add to the playlist",
    on_change=search_songs,
    key="search_input",
)

search_results = st.session_state.search_results
playlist_songs = st.session_state.playlist_songs
recommendations = st.session_state.recommendations

with col_res:
    if search_results:
        st.subheader("Search Results")
    for song_id, song in search_results:
        col_text, col_button = st.columns([4, 1])
        col_text.markdown(f"{song}")
        col_button.button(
            "&#43;",
            key=f"add_{song_id}",
            on_click=add_to_playlist,
            args=(song_id, song),
        )

with col_list:
    st.subheader("Playlist")
    for i, (song_id, song) in enumerate(playlist_songs):
        col_text, col_button = st.columns([3, 1])
        col_text.markdown(f"{i + 1}. {song}")
        col_button.button(
            "&#45;",
            key=f"remove_{i}",
            on_click=remove_from_playlist,
            args=(i,),
        )

with col_recsys:
    st.subheader("Recommendations")
    if playlist_songs:
        song_ids = [int(song_id) for song_id, _ in playlist_songs]
        song_ids_str = ", ".join(str(song_id) for song_id in song_ids)
        recommendations = st.session_state.get("recommendations", None)
        prompt = st.session_state.get("prompt", "")
        col, _ = st.columns([2, 1])

        with col:
            st.button(
                "Get recommendations",
                on_click=get_recommendations,
                args=(song_ids,),
                use_container_width=True,
            )
            if recommendations:
                with st.popover("View prompt", use_container_width=True):
                    st.markdown(prompt.replace("\n", "\n\n"))
                for i, rec in enumerate(recommendations):
                    uri = lookup_spotify_uri(rec)
                    if uri:
                        components.html(
                            f"""
                            <script src="https://open.spotify.com/embed/iframe-api/v1" async></script>
                            <div id="iframe-rec-{i}"></div>
                            <script>
                            window.onSpotifyIframeApiReady = (IFrameAPI) => {{
                                const element = document.getElementById('iframe-rec-{i}');
                                const options = {{
                                    width: '100%',
                                    height: '80',
                                    uri: '{uri}'
                                }};
                                const callback = (EmbedController) => {{}};
                                IFrameAPI.createController(element, options, callback);
                            }};
                            </script>
                            """,
                            height=88,
                            scrolling=False,
                        )
                    else:
                        st.write(f"{rec}")
    else:
        st.write("Add songs to the playlist to get recommendations.")
