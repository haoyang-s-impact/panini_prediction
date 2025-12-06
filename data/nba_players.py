"""
NBA Player Names and Mappings for OCR Feature Extraction

This module contains:
- Top 300 NBA players (2018-2025 seasons)
- Chinese-to-English player name mappings (top 50-100 players)
- NBA team name mappings (Chinese/English → full team names)
- Panini parallel color mappings
- Card descriptor keyword mappings
"""

# Top 300 NBA Players (2018-2025) - Most valuable for trading cards
NBA_PLAYERS = [
    # Current Superstars & MVPs
    "Stephen Curry", "LeBron James", "Kevin Durant", "Giannis Antetokounmpo",
    "Nikola Jokic", "Joel Embiid", "Luka Doncic", "Kawhi Leonard",
    "Damian Lillard", "Anthony Davis", "James Harden", "Kyrie Irving",

    # Rising Stars & Top Rookies (2018-2024)
    "Ja Morant", "Zion Williamson", "Shai Gilgeous-Alexander", "Jayson Tatum",
    "Trae Young", "Donovan Mitchell", "Devin Booker", "Anthony Edwards",
    "LaMelo Ball", "Tyrese Haliburton", "Paolo Banchero", "Victor Wembanyama",
    "Chet Holmgren", "Scoot Henderson", "Brandon Miller", "Amen Thompson",

    # All-Stars & Key Players
    "Jaylen Brown", "Jimmy Butler", "Paul George", "Bam Adebayo",
    "Karl-Anthony Towns", "DeMar DeRozan", "Bradley Beal", "Klay Thompson",
    "Draymond Green", "Rudy Gobert", "Khris Middleton", "Jrue Holiday",
    "Chris Paul", "Russell Westbrook", "Kristaps Porzingis", "Pascal Siakam",

    # High-Value Rookies & Sophomores
    "Cade Cunningham", "Scottie Barnes", "Evan Mobley", "Jalen Green",
    "Franz Wagner", "Alperen Sengun", "Jalen Suggs", "Josh Giddey",
    "Herb Jones", "Ayo Dosunmu", "Bones Hyland", "Scottie Pippen Jr",

    # 2018-2020 Rookie Class (High Value)
    "Luka Doncic", "Trae Young", "Jaren Jackson Jr", "Marvin Bagley III",
    "Wendell Carter Jr", "Collin Sexton", "Shai Gilgeous-Alexander",
    "Miles Bridges", "Kevin Huerter", "Jalen Brunson", "De'Anthony Melton",
    "Landry Shamet", "Gary Trent Jr", "Robert Williams III",

    # 2019-2020 Class
    "Zion Williamson", "Ja Morant", "RJ Barrett", "De'Andre Hunter",
    "Darius Garland", "Coby White", "Cam Reddish", "Jarrett Culver",
    "Rui Hachimura", "Jaxson Hayes", "Tyler Herro", "Romeo Langford",
    "Keldon Johnson", "Brandon Clarke", "PJ Washington", "Jordan Poole",

    # Additional Stars & Veterans
    "Kawhi Leonard", "Paul George", "Tobias Harris", "Al Horford",
    "Marcus Smart", "Malcolm Brogdon", "Terry Rozier", "CJ McCollum",
    "Fred VanVleet", "OG Anunoby", "Dejounte Murray", "De'Aaron Fox",
    "Domantas Sabonis", "Julius Randle", "Zach LaVine", "Nikola Vucevic",

    # Warriors Dynasty Players
    "Stephen Curry", "Klay Thompson", "Draymond Green", "Andrew Wiggins",
    "Jordan Poole", "James Wiseman", "Jonathan Kuminga", "Moses Moody",

    # Lakers Core
    "LeBron James", "Anthony Davis", "Austin Reaves", "Rui Hachimura",
    "D'Angelo Russell", "Jarred Vanderbilt",

    # Nets/76ers Stars
    "Joel Embiid", "James Harden", "Tyrese Maxey", "Tobias Harris",
    "Kevin Durant", "Kyrie Irving", "Ben Simmons",

    # Bucks Core
    "Giannis Antetokounmpo", "Damian Lillard", "Khris Middleton", "Brook Lopez",
    "Jrue Holiday",

    # Nuggets Championship Core
    "Nikola Jokic", "Jamal Murray", "Michael Porter Jr", "Aaron Gordon",
    "Kentavious Caldwell-Pope",

    # Mavericks
    "Luka Doncic", "Kyrie Irving", "Tim Hardaway Jr", "Christian Wood",
    "Jaden Hardy",

    # Grizzlies Young Core
    "Ja Morant", "Jaren Jackson Jr", "Desmond Bane", "Brandon Clarke",
    "Ziaire Williams", "GG Jackson",

    # Thunder Rising Stars
    "Shai Gilgeous-Alexander", "Chet Holmgren", "Josh Giddey", "Jalen Williams",
    "Luguentz Dort", "Cason Wallace",

    # Pelicans Core
    "Zion Williamson", "Brandon Ingram", "CJ McCollum", "Herbert Jones",
    "Trey Murphy III", "Dyson Daniels",

    # Timberwolves
    "Anthony Edwards", "Karl-Anthony Towns", "Rudy Gobert", "Mike Conley",
    "Jaden McDaniels", "Naz Reid",

    # Kings Resurgence
    "De'Aaron Fox", "Domantas Sabonis", "Keegan Murray", "Kevin Huerter",
    "Malik Monk",

    # Suns Big 3
    "Kevin Durant", "Devin Booker", "Bradley Beal", "Jusuf Nurkic",

    # Clippers
    "Kawhi Leonard", "Paul George", "Russell Westbrook", "Norman Powell",
    "Terance Mann",

    # Heat Culture
    "Jimmy Butler", "Bam Adebayo", "Tyler Herro", "Kyle Lowry",
    "Duncan Robinson", "Caleb Martin",

    # Celtics Championship Contenders
    "Jayson Tatum", "Jaylen Brown", "Kristaps Porzingis", "Jrue Holiday",
    "Derrick White", "Al Horford",

    # Cavaliers Young Core
    "Donovan Mitchell", "Darius Garland", "Evan Mobley", "Jarrett Allen",
    "Caris LeVert", "Isaac Okoro",

    # Knicks Rebuild
    "Jalen Brunson", "Julius Randle", "RJ Barrett", "Mitchell Robinson",
    "Immanuel Quickley", "Quentin Grimes",

    # Hawks
    "Trae Young", "Dejounte Murray", "De'Andre Hunter", "Clint Capela",
    "AJ Griffin", "Jalen Johnson",

    # Magic Young Talent
    "Paolo Banchero", "Franz Wagner", "Jalen Suggs", "Wendell Carter Jr",
    "Cole Anthony", "Markelle Fultz",

    # Pacers Speed
    "Tyrese Haliburton", "Myles Turner", "Bennedict Mathurin", "Buddy Hield",
    "TJ McConnell", "Obi Toppin",

    # Raptors
    "Scottie Barnes", "Pascal Siakam", "OG Anunoby", "Gary Trent Jr",
    "Jakob Poeltl", "Gradey Dick",

    # Bulls
    "Zach LaVine", "DeMar DeRozan", "Nikola Vucevic", "Coby White",
    "Patrick Williams", "Ayo Dosunmu",

    # Nets Rebuild
    "Mikal Bridges", "Cam Thomas", "Nic Claxton", "Ben Simmons",
    "Cam Johnson",

    # Hornets
    "LaMelo Ball", "Brandon Miller", "Mark Williams", "Miles Bridges",
    "Terry Rozier", "Nick Richards",

    # Pistons Rebuild
    "Cade Cunningham", "Jaden Ivey", "Ausar Thompson", "Jalen Duren",
    "Isaiah Stewart", "Marcus Sasser",

    # Rockets Young Core
    "Jalen Green", "Alperen Sengun", "Jabari Smith Jr", "Amen Thompson",
    "Tari Eason", "Cam Whitmore",

    # Spurs Wembanyama Era
    "Victor Wembanyama", "Devin Vassell", "Keldon Johnson", "Jeremy Sochan",
    "Tre Jones", "Malaki Branham",

    # Blazers Rebuild
    "Scoot Henderson", "Shaedon Sharpe", "Anfernee Simons", "Jerami Grant",
    "Deandre Ayton", "Malcolm Brogdon",

    # Jazz Rebuild
    "Lauri Markkanen", "Walker Kessler", "Keyonte George", "Collin Sexton",
    "Jordan Clarkson", "John Collins",

    # Wizards Rebuild
    "Jordan Poole", "Kyle Kuzma", "Tyus Jones", "Deni Avdija",
    "Bilal Coulibaly", "Corey Kispert",
]

# Chinese to English Player Name Mappings (Top 50-100 Players)
CHINESE_NAME_MAP = {
    # Superstars
    "史蒂芬库里": "Stephen Curry",
    "库里": "Stephen Curry",
    "勒布朗詹姆斯": "LeBron James",
    "詹姆斯": "LeBron James",
    "勒布朗": "LeBron James",
    "凯文杜兰特": "Kevin Durant",
    "杜兰特": "Kevin Durant",
    "字母哥": "Giannis Antetokounmpo",
    "扬尼斯": "Giannis Antetokounmpo",
    "约基奇": "Nikola Jokic",
    "恩比德": "Joel Embiid",
    "东契奇": "Luka Doncic",
    "卢卡": "Luka Doncic",
    "伦纳德": "Kawhi Leonard",
    "小卡": "Kawhi Leonard",

    # Rising Stars
    "贾莫兰特": "Ja Morant",
    "莫兰特": "Ja Morant",
    "锡安": "Zion Williamson",
    "威廉姆森": "Zion Williamson",
    "亚历山大": "Shai Gilgeous-Alexander",
    "吉尔杰斯亚历山大": "Shai Gilgeous-Alexander",
    "SGA": "Shai Gilgeous-Alexander",
    "塔图姆": "Jayson Tatum",
    "特雷杨": "Trae Young",
    "杨": "Trae Young",
    "布克": "Devin Booker",
    "爱德华兹": "Anthony Edwards",
    "拉梅洛鲍尔": "LaMelo Ball",
    "鲍尔": "LaMelo Ball",
    "哈利伯顿": "Tyrese Haliburton",
    "班凯罗": "Paolo Banchero",
    "文班亚马": "Victor Wembanyama",
    "温班亚马": "Victor Wembanyama",
    "切特": "Chet Holmgren",

    # All-Stars
    "米切尔": "Donovan Mitchell",
    "巴特勒": "Jimmy Butler",
    "保罗乔治": "Paul George",
    "PG13": "Paul George",
    "阿德巴约": "Bam Adebayo",
    "唐斯": "Karl-Anthony Towns",
    "德罗赞": "DeMar DeRozan",
    "比尔": "Bradley Beal",
    "克莱汤普森": "Klay Thompson",
    "汤普森": "Klay Thompson",
    "追梦格林": "Draymond Green",
    "格林": "Draymond Green",
    "戈贝尔": "Rudy Gobert",
    "米德尔顿": "Khris Middleton",
    "霍勒迪": "Jrue Holiday",
    "保罗": "Chris Paul",
    "CP3": "Chris Paul",
    "威少": "Russell Westbrook",
    "威斯布鲁克": "Russell Westbrook",

    # High-Value Rookies
    "坎宁安": "Cade Cunningham",
    "巴恩斯": "Scottie Barnes",
    "莫布里": "Evan Mobley",
    "格林": "Jalen Green",
    "瓦格纳": "Franz Wagner",
    "森gun": "Alperen Sengun",
    "吉迪": "Josh Giddey",

    # Other Notable
    "穆雷": "Jamal Murray",
    "波特": "Michael Porter Jr",
    "班恩": "Desmond Bane",
    "杰克逊": "Jaren Jackson Jr",
    "英格拉姆": "Brandon Ingram",
    "福克斯": "De'Aaron Fox",
    "萨博尼斯": "Domantas Sabonis",
    "兰德尔": "Julius Randle",
    "拉文": "Zach LaVine",
    "武切维奇": "Nikola Vucevic",
}

# NBA Team Name Mappings (Chinese/English → Full Team Names)
TEAM_MAPPINGS = {
    # Warriors
    "勇士": "Golden State Warriors",
    "WARRIORS": "Golden State Warriors",
    "Warriors": "Golden State Warriors",
    "GSW": "Golden State Warriors",
    "金州勇士": "Golden State Warriors",

    # Lakers
    "湖人": "Los Angeles Lakers",
    "LAKERS": "Los Angeles Lakers",
    "Lakers": "Los Angeles Lakers",
    "LAL": "Los Angeles Lakers",
    "洛杉矶湖人": "Los Angeles Lakers",

    # Celtics
    "凯尔特人": "Boston Celtics",
    "CELTICS": "Boston Celtics",
    "Celtics": "Boston Celtics",
    "BOS": "Boston Celtics",

    # Heat
    "热火": "Miami Heat",
    "HEAT": "Miami Heat",
    "Heat": "Miami Heat",
    "MIA": "Miami Heat",

    # Bucks
    "雄鹿": "Milwaukee Bucks",
    "BUCKS": "Milwaukee Bucks",
    "Bucks": "Milwaukee Bucks",
    "MIL": "Milwaukee Bucks",

    # Grizzlies
    "灰熊": "Memphis Grizzlies",
    "GRIZZLIES": "Memphis Grizzlies",
    "Grizzlies": "Memphis Grizzlies",
    "MEM": "Memphis Grizzlies",

    # Clippers
    "快船": "Los Angeles Clippers",
    "CLIPPERS": "Los Angeles Clippers",
    "Clippers": "Los Angeles Clippers",
    "LAC": "Los Angeles Clippers",

    # Nets
    "篮网": "Brooklyn Nets",
    "NETS": "Brooklyn Nets",
    "Nets": "Brooklyn Nets",
    "BKN": "Brooklyn Nets",

    # 76ers
    "76人": "Philadelphia 76ers",
    "SIXERS": "Philadelphia 76ers",
    "76ERS": "Philadelphia 76ers",
    "PHI": "Philadelphia 76ers",

    # Nuggets
    "掘金": "Denver Nuggets",
    "NUGGETS": "Denver Nuggets",
    "Nuggets": "Denver Nuggets",
    "DEN": "Denver Nuggets",

    # Mavericks
    "独行侠": "Dallas Mavericks",
    "MAVERICKS": "Dallas Mavericks",
    "Mavericks": "Dallas Mavericks",
    "DAL": "Dallas Mavericks",

    # Suns
    "太阳": "Phoenix Suns",
    "SUNS": "Phoenix Suns",
    "Suns": "Phoenix Suns",
    "PHX": "Phoenix Suns",

    # Thunder
    "雷霆": "Oklahoma City Thunder",
    "THUNDER": "Oklahoma City Thunder",
    "Thunder": "Oklahoma City Thunder",
    "OKC": "Oklahoma City Thunder",

    # Pelicans
    "鹈鹕": "New Orleans Pelicans",
    "PELICANS": "New Orleans Pelicans",
    "Pelicans": "New Orleans Pelicans",
    "NOP": "New Orleans Pelicans",

    # Timberwolves
    "森林狼": "Minnesota Timberwolves",
    "TIMBERWOLVES": "Minnesota Timberwolves",
    "Timberwolves": "Minnesota Timberwolves",
    "MIN": "Minnesota Timberwolves",

    # Kings
    "国王": "Sacramento Kings",
    "KINGS": "Sacramento Kings",
    "Kings": "Sacramento Kings",
    "SAC": "Sacramento Kings",

    # Cavaliers
    "骑士": "Cleveland Cavaliers",
    "CAVALIERS": "Cleveland Cavaliers",
    "Cavaliers": "Cleveland Cavaliers",
    "CLE": "Cleveland Cavaliers",

    # Knicks
    "尼克斯": "New York Knicks",
    "KNICKS": "New York Knicks",
    "Knicks": "New York Knicks",
    "NYK": "New York Knicks",

    # Hawks
    "老鹰": "Atlanta Hawks",
    "HAWKS": "Atlanta Hawks",
    "Hawks": "Atlanta Hawks",
    "ATL": "Atlanta Hawks",

    # Magic
    "魔术": "Orlando Magic",
    "MAGIC": "Orlando Magic",
    "Magic": "Orlando Magic",
    "ORL": "Orlando Magic",

    # Pacers
    "步行者": "Indiana Pacers",
    "PACERS": "Indiana Pacers",
    "Pacers": "Indiana Pacers",
    "IND": "Indiana Pacers",

    # Raptors
    "猛龙": "Toronto Raptors",
    "RAPTORS": "Toronto Raptors",
    "Raptors": "Toronto Raptors",
    "TOR": "Toronto Raptors",

    # Bulls
    "公牛": "Chicago Bulls",
    "BULLS": "Chicago Bulls",
    "Bulls": "Chicago Bulls",
    "CHI": "Chicago Bulls",

    # Hornets
    "黄蜂": "Charlotte Hornets",
    "HORNETS": "Charlotte Hornets",
    "Hornets": "Charlotte Hornets",
    "CHA": "Charlotte Hornets",

    # Pistons
    "活塞": "Detroit Pistons",
    "PISTONS": "Detroit Pistons",
    "Pistons": "Detroit Pistons",
    "DET": "Detroit Pistons",

    # Rockets
    "火箭": "Houston Rockets",
    "ROCKETS": "Houston Rockets",
    "Rockets": "Houston Rockets",
    "HOU": "Houston Rockets",

    # Spurs
    "马刺": "San Antonio Spurs",
    "SPURS": "San Antonio Spurs",
    "Spurs": "San Antonio Spurs",
    "SAS": "San Antonio Spurs",

    # Trail Blazers
    "开拓者": "Portland Trail Blazers",
    "BLAZERS": "Portland Trail Blazers",
    "Trail Blazers": "Portland Trail Blazers",
    "POR": "Portland Trail Blazers",

    # Jazz
    "爵士": "Utah Jazz",
    "JAZZ": "Utah Jazz",
    "Jazz": "Utah Jazz",
    "UTA": "Utah Jazz",

    # Wizards
    "奇才": "Washington Wizards",
    "WIZARDS": "Washington Wizards",
    "Wizards": "Washington Wizards",
    "WAS": "Washington Wizards",
}

# Panini Parallel Color Mappings
PARALLEL_MAPPINGS = {
    "银折": "Silver",
    "紫折": "Purple",
    "红碎冰": "Red Ice",
    "红折": "Red",
    "蓝折": "Blue",
    "绿": "Green",
    "绿折": "Green",
    "金": "Gold",
    "金折": "Gold",
    "粉碎冰": "Pink Ice",
    "粉折": "Pink",
    "橙折": "Orange",
    "黑": "Black",
    "白": "White",
    "彩虹": "Rainbow",
    "斑马": "Zebra",
    "迷彩": "Camo",
    "SILVER": "Silver",
    "PURPLE": "Purple",
    "RED ICE": "Red Ice",
    "REDICE": "Red Ice",
    "BLUE": "Blue",
    "GREEN": "Green",
    "GOLD": "Gold",
    "PINK": "Pink",
    "ORANGE": "Orange",
    "BLACK": "Black",
    "RAINBOW": "Rainbow",
}

# Card Descriptor Keyword Mappings
DESCRIPTOR_KEYWORDS = {
    'autograph': [
        r'签字', r'签名', r'卡签', r'Auto(?:graph)?', r'Signed', r'签',
    ],
    'rookie': [
        r'\bRC\b', r'新秀', r'Rookie', r'ROOKIE', r'新人',
    ],
    'patch': [
        r'球衣', r'Patch', r'PATCH', r'Jersey', r'JERSEY', r'切割',
    ],
    'refractor': [
        r'折射', r'Refractor', r'REFRACTOR', r'折',
    ],
    'rpa': [
        r'\bRPA\b', r'Rookie Patch Auto',
    ],
    'memorabilia': [
        r'实物', r'Memorabilia', r'Game[- ]Used',
    ],
    'numbered': [
        r'编号', r'Numbered', r'/\d+', r'编',
    ],
    'prizm': [
        r'Prizm', r'PRIZM', r'棱镜',
    ],
    'base': [
        r'\bBase\b', r'BASE', r'基础',
    ],
}
