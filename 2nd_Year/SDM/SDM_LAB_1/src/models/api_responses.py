
paper_bulk_response : dict = {
  "total": 0,
  "token": "some_token_for_pagination",
  "data": [
    {
      "paperId": "some_paper_id",
      "title": "some_paper_title",
    }
  ]
}

paper_details_response : dict = {
  "paperId": "paper_id",
  "title": "Paper Title",
  "abstract": "Abstract text.",
  "venue": "Venue",
  "publicationVenue": {
    "id": "venue_id",
    "name": "venue_name",
    "type": "venue_type",
    "alternate_names": [
      "List",
      "of",
      "alternative",
      "Venue names"
    ],
    "url": "https://venue_url.com"
  },
  "year": 1969,
  "fieldsOfStudy": [
    "field_of_study_name"
  ],
  "publicationDate": "YYY-MM-DD",
  "journal": {
    "volume": "XYZ",
    "pages": "UVW - XYZ",
    "name": "publication_name"
  },
  "authors": [
    {
      "authorId": "author_id",
      "url": "https://www.semanticscholar.org/author/author_id",
      "name": "author_name",
      "affiliations": [
        "affiliation_name"
      ],
      "homepage": "author_homepage_url",
      "paperCount": 0,
      "citationCount": 0,
      "hIndex": 0
    }
  ],
  "references": [
    {
      "paperId": "5c5751d45e298cea054f32b392c12c61027d2fe7"
    }
  ],
}
