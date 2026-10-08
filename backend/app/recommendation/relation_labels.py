"""Vietnamese labels for Freebase relation names used by the three knowledge graphs.

A relation such as ``film.film.set_designer`` points from an item to a tail entity whose
role is named by the last segment, so that segment decides both the wording of a reason
("cùng thiết kế bối cảnh với ...") and the label of an entity that has no curated name.
"""
from typing import Dict

_ROLE_LABELS: Dict[str, str] = {
    # People behind a film
    "actor": "diễn viên",
    "star": "diễn viên chính",
    "regular_cast": "dàn diễn viên",
    "director": "đạo diễn",
    "writer": "biên kịch",
    "producer": "nhà sản xuất",
    "executive_producer": "giám đốc sản xuất",
    "cinematographer": "quay phim",
    "art_director": "giám đốc mỹ thuật",
    "casting_director": "giám đốc tuyển vai",
    "costume_designer": "thiết kế phục trang",
    "production_designer": "thiết kế sản xuất",
    "set_designer": "thiết kế bối cảnh",
    "music": "nhạc sĩ phim",
    "production_company": "hãng sản xuất",
    "performance": "vai diễn",
    "starring_roles": "vai diễn truyền hình",
    # Film attributes
    "genre": "thể loại",
    "genre_ranking": "xếp hạng thể loại",
    "country": "quốc gia",
    "country_of_origin": "quốc gia",
    "nationality": "quốc tịch",
    "language": "ngôn ngữ",
    "location": "bối cảnh quay",
    "rating": "phân loại độ tuổi",
    "sequel": "phần tiếp theo",
    "series": "loạt phim",
    "film": "bộ phim",
    # Awards
    "nomination": "đề cử giải thưởng",
    "nominations": "đề cử giải thưởng",
    "award_nominations": "đề cử giải thưởng",
    "honor": "giải thưởng",
    "honors": "giải thưởng",
    "awards_won": "giải thưởng",
    # Books
    "author": "tác giả",
    "works_written": "tác phẩm",
    "works_edited": "tác phẩm biên tập",
    "character": "nhân vật",
    "appears_in_book": "cuốn sách",
    "book": "cuốn sách",
    "interior_illustrations_by": "hoạ sĩ minh hoạ",
    "publisher": "nhà xuất bản",
    "date_of_first_publication": "năm xuất bản đầu",
    "date_written": "năm sáng tác",
    "subject": "chủ đề",
    "translation": "bản dịch",
    "literary_series": "bộ sách",
    "works_in_this_series": "bộ sách",
    "series_written_or_contributed_to": "bộ sách",
    "previous_in_series": "tập trước trong bộ",
    "number_of_volumes": "số tập",
    "comic_book_series_published": "bộ truyện tranh",
    "epub": "ấn bản điện tử",
    "kindle": "ấn bản điện tử",
    "pdf": "ấn bản điện tử",
    "text": "ấn bản điện tử",
    # Music and people
    "artist": "nghệ sĩ",
    "album": "album",
    "track": "bài hát",
    "tracks_produced": "bài hát đã sản xuất",
    "instruments_played": "nhạc cụ",
    "origin": "nơi xuất thân",
    "place_of_birth": "nơi sinh",
    "place_of_death": "nơi mất",
    "profession": "nghề nghiệp",
    "creator": "tác giả nhân vật",
    "industry": "ngành",
    "games_published": "trò chơi",
    "contained_by": "khu vực",
    "time_zone": "múi giờ",
    "key": "mã định danh",
    "subject_key": "mã định danh",
}

# Roles the frontend draws with their own colour; everything else is a generic "Entity".
_NODE_TYPES: Dict[str, str] = {
    "director": "Director",
    "actor": "Actor",
    "star": "Actor",
    "regular_cast": "Actor",
    "genre": "Genre",
    "writer": "Writer",
    "author": "Writer",
    "producer": "Producer",
    "executive_producer": "Producer",
    "production_company": "Producer",
}


def _role_key(raw_relation: str) -> str:
    return (raw_relation or "").strip().lower().split(".")[-1]


def relation_role(raw_relation: str) -> str:
    """Vietnamese noun for the tail entity of a relation, e.g. 'film.film.director' -> 'đạo diễn'."""
    key = _role_key(raw_relation)
    raw_lower = (raw_relation or "").lower()
    if key == "editor":
        return "biên tập viên" if raw_lower.startswith("book.") else "dựng phim"
    if key == "series" and raw_lower.startswith("comic_books."):
        return "bộ truyện tranh"
    if key == "publisher" and not raw_lower.startswith(("book.", "comic_books.")):
        return "nhà phát hành"
    if key in _ROLE_LABELS:
        return _ROLE_LABELS[key]
    return key.replace("_", " ") or "thực thể tri thức"


def relation_node_type(raw_relation: str) -> str:
    return _NODE_TYPES.get(_role_key(raw_relation), "Entity")
