from typing import List, Tuple


def get_playlist(playlist_name: str) -> List[Tuple[int, str]]:
    if playlist_name == "Rock":
        return rock_playlist
    elif playlist_name == "Pop":
        return pop_playlist
    elif playlist_name == "Rap":
        return rap_playlist
    elif playlist_name == "Spanish":
        return spanish_playlist
    else:
        return []


rock_playlist = [
    ("28968", "Bohemian Rhapsody, Queen"),
    ("11063", "November Rain, Guns N' Roses"),
    ("8901", "Hotel California, Eagles"),
    ("27869", "Sweet Child O' Mine, Guns N' Roses"),
    ("7681", "Smoke On The Water, Deep Purple"),
    ("13047", "Back In Black, Ac/Dc"),
    ("29496", "Free Bird, Lynyrd Skynyrd"),
    ("31312", "Born To Run, Bruce Springsteen"),
    ("6601", "Dream On, Aerosmith"),
    ("15585", "Comfortably Numb, Pink Floyd"),
    ("29494", "Layla, Derek And The Dominos"),
    ("7878", "More Than A Feeling, Boston"),
    ("15872", "Rock You Like A Hurricane, Scorpions"),
    ("26098", "Sweet Home Alabama, Lynyrd Skynyrd"),
    ("17836", "Baba O'Riley, The Who"),
    ("13823", "Another Brick In The Wall Part, Pink Floyd"),
    ("5632", "Purple Haze, Jimi Hendrix"),
    ("27590", "Go Your Own Way, Fleetwood Mac"),
    ("16332", "Carry On Wayward Son, Kansas"),
    ("8816", "Highway To Hell, Ac/Dc"),
]

pop_playlist = [
    ("12733", "Billie Jean, Michael Jackson"),
    ("4761", "Like A Prayer, Madonna"),
    ("4683", "I Want It That Way, Backstreet Boys"),
    ("62104", "Baby One More Time, Britney Spears"),
    ("196", "Viva La Vida, Coldplay"),
    ("25830", "Poker Face, Lady Gaga"),
    ("24093", "Toxic, Britney Spears"),
    ("29937", "Since U Been Gone, Kelly Clarkson"),
    ("2623", "Rehab, Amy Winehouse"),
    ("7255", "Oops!...I Did It Again, Britney Spears"),
    ("16362", "Hot N Cold, Katy Perry"),
    ("64889", "Irreplaceable, Beyoncé"),
    ("25778", "Complicated, Avril Lavigne"),
    ("19145", "It'S Gonna Be Me, *Nsync"),
    ("39703", "Cant Get You Out Of My Head, Kylie Minogue"),
    ("3021", "Mr Brightside, The Killers"),
    ("6628", "Beautiful Day, U2"),
    ("26029", "Genie In A Bottle, Christina Aguilera"),
    ("24132", "Hollaback Girl, Gwen Stefani"),
    ("25975", "Unwritten, Natasha Bedingfield"),
]

rap_playlist = [
    ("15030", "Juicy, The Notorious B.I.G."),
    ("20077", "Hypnotize, The Notorious B.I.G."),
    ("19612", "C.R.E.A.M, Wu-Tang Clan"),
    ("49669", "Gin And Juice, Snoop Dogg"),
    ("13032", "Fight The Power, Public Enemy"),
    ("7247", "Ms Jackson, Outkast"),
    ("49487", "Dear Mama, 2Pac"),
    ("51765", "Get Ur Freak On, Missy Elliott"),
    ("49079", "Rosa Parks, Outkast"),
    ("16795", "It Was A Good Day, Ice Cube"),
    ("45773", "In Da Club, 50 Cent"),
    ("62463", "Hot In Herre, Nelly"),
    ("28422", "My Name Is, Eminem"),
    ("49356", "The Real Slim Shady, Eminem"),
    ("7249", "Hey Ya!, Outkast"),
    ("51900", "Mama Said Knock You Out, Ll Cool J"),
    ("49479", "Changes, 2Pac"),
    ("19542", "Ice Ice Baby, Vanilla Ice"),
    ("16815", "99 Problems, Jay-Z"),
    ("12834", "U Can'T Touch This, Mc Hammer"),
]

spanish_playlist = [
    ("33375", "La Bamba, Ritchie Valens"),
    ("12855", "Macarena, Los Del Río"),
    ("14745", "Livin' La Vida Loca, Ricky Martin"),
    ("73502", "Bailamos, Enrique Iglesias"),
    ("55643", "Oye Como Va, Santana"),
    ("34855", "La Camisa Negra, Juanes"),
    ("62322", "Ciega, Sordomuda, Shakira"),
    ("62042", "Gasolina, Daddy Yankee"),
    ("10495", "Corazón Partío, Alejandro Sanz"),
    ("58037", "Burbujas De Amor, Juan Luis Guerra"),
    ("59220", "La Flaca, Jarabe De Palo"),
    ("59216", "Maldito Duende, Héroes Del Silencio"),
    ("66047", "A Dios Le Pido, Juanes"),
    ("47002", "Pienso En Ti, Shakira"),
    ("62821", "Me Voy, Julieta Venegas"),
    ("45078", "Tenía Tanto Que Darte, Nena Daconte"),
    ("14156", "Que Te Quería, La Quinta Estación"),
    ("63725", "El Universo Sobre Mi, Amaral"),
    ("55679", "Entre Dos Aguas, Paco De Lucía"),
    ("40014", "Entre Dos Tierras, Héroes Del Silencio"),
]
