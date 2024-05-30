def load_playlist_map(dataset_map: dict[int, str]) -> dict[str, list[tuple[int, str]]]:
    playlist_map = dict[str, list[tuple[int, str]]]()
    playlist_map["Rock"] = find_ids(dataset_map, rock_playlist)
    playlist_map["Pop"] = find_ids(dataset_map, pop_playlist)
    playlist_map["Rap"] = find_ids(dataset_map, rap_playlist)
    playlist_map["Spanish"] = find_ids(dataset_map, spanish_playlist)
    return playlist_map


def find_ids(dataset_map: dict[int, str], playlist: list[str]) -> list[tuple[int, str]]:
    matching_keys = list[tuple[int, str]]()
    for song_title in playlist:
        found_key = None
        for key, value in dataset_map.items():
            if song_title in value:
                found_key = key
                break
        if found_key is not None:
            matching_keys.append((found_key, song_title))
    return matching_keys


rock_playlist = [
    "Bohemian Rhapsody, Queen",
    "November Rain, Guns N' Roses",
    "Hotel California, Eagles",
    "Sweet Child O' Mine, Guns N' Roses",
    "Smoke On The Water, Deep Purple",
    "Back In Black, Ac/Dc",
    "Free Bird, Lynyrd Skynyrd",
    "Born To Run, Bruce Springsteen",
    "Dream On, Aerosmith",
    "Comfortably Numb, Pink Floyd",
    "Layla, Derek And The Dominos",
    "More Than A Feeling, Boston",
    "Rock You Like A Hurricane, Scorpions",
    "Sweet Home Alabama, Lynyrd Skynyrd",
    "Baba O'Riley, The Who",
    "Another Brick In The Wall Part, Pink Floyd",
    "Purple Haze, Jimi Hendrix",
    "Go Your Own Way, Fleetwood Mac",
    "Carry On Wayward Son, Kansas",
    "Highway To Hell, Ac/Dc",
]

pop_playlist = [
    "Billie Jean, Michael Jackson",
    "Like A Prayer, Madonna",
    "I Want It That Way, Backstreet Boys",
    "Baby One More Time, Britney Spears",
    "Viva La Vida, Coldplay",
    "Poker Face, Lady Gaga",
    "Toxic, Britney Spears",
    "Since U Been Gone, Kelly Clarkson",
    "Rehab, Amy Winehouse",
    "Oops!...I Did It Again, Britney Spears",
    "Hot N Cold, Katy Perry",
    "Irreplaceable, Beyoncé",
    "Complicated, Avril Lavigne",
    "It'S Gonna Be Me, *Nsync",
    "Cant Get You Out Of My Head, Kylie Minogue",
    "Mr Brightside, The Killers",
    "Beautiful Day, U2",
    "Genie In A Bottle, Christina Aguilera",
    "Hollaback Girl, Gwen Stefani",
    "Unwritten, Natasha Bedingfield",
]

rap_playlist = [
    "Juicy, The Notorious B.I.G.",
    "Hypnotize, The Notorious B.I.G.",
    "C.R.E.A.M, Wu-Tang Clan",
    "Gin And Juice, Snoop Dogg",
    "Fight The Power, Public Enemy",
    "Ms Jackson, Outkast",
    "Dear Mama, 2Pac",
    "Get Ur Freak On, Missy Elliott",
    "Rosa Parks, Outkast",
    "It Was A Good Day, Ice Cube",
    "In Da Club, 50 Cent",
    "Hot In Herre, Nelly",
    "My Name Is, Eminem",
    "The Real Slim Shady, Eminem",
    "Hey Ya!, Outkast",
    "Mama Said Knock You Out, Ll Cool J",
    "Changes, 2Pac",
    "Ice Ice Baby, Vanilla Ice",
    "99 Problems, Jay-Z",
    "U Can'T Touch This, Mc Hammer",
]

spanish_playlist = [
    "La Bamba, Ritchie Valens",
    "Macarena, Los Del Río",
    "Livin' La Vida Loca, Ricky Martin",
    "Bailamos, Enrique Iglesias",
    "Oye Como Va, Santana",
    "La Camisa Negra, Juanes",
    "Ciega, Sordomuda, Shakira",
    "Gasolina, Daddy Yankee",
    "Corazón Partío, Alejandro Sanz",
    "Burbujas De Amor, Juan Luis Guerra",
    "La Flaca, Jarabe De Palo",
    "Maldito Duende, Héroes Del Silencio",
    "A Dios Le Pido, Juanes",
    "Pienso En Ti, Shakira",
    "Me Voy, Julieta Venegas",
    "Tenía Tanto Que Darte, Nena Daconte",
    "Que Te Quería, La Quinta Estación",
    "El Universo Sobre Mi, Amaral",
    "Entre Dos Aguas, Paco De Lucía",
    "Entre Dos Tierras, Héroes Del Silencio",
]
