from enum import IntEnum, unique
from dataclasses import dataclass

@dataclass(frozen=True)
class ObjectClassification:
    eigen13: int
    rio27: int
    nyu40: int
    rio: int  # the full 528 rio class
    label: str

@unique
class Objects3DSSG(IntEnum):
    def __str__(self):
        return super().__str__().replace("Objects","")
    air_conditioner = 0
    apron = 1
    aquarium = 2
    armchair = 3
    armoire = 4
    armor = 5
    audio_system = 6
    baby_bed = 7
    baby_changing_table = 8
    baby_changing_unit = 9
    baby_gym = 10
    baby_seat = 11
    baby_toys = 12
    backpack = 13
    bag = 14
    balcony = 15
    balcony_door = 16
    ball = 17
    bar = 18
    bar_stool = 19
    barrel = 20
    barstool = 21
    basin = 22
    basket = 23
    bath_cabinet = 24
    bath_counter = 25
    bath_rack = 26
    bath_robe = 27
    bathrobe = 28
    bathroom_items = 29
    bathtub = 30
    bbq = 31
    beam = 32
    bean_bag = 33
    beanbag = 34
    beautician = 35
    bed = 36
    bed_table = 37
    bedside_table = 38
    bench = 39
    beverage_crate = 40
    bicycle = 41
    bidet = 42
    bike = 43
    bin = 44
    blackboard = 45
    blanket = 46
    blinds = 47
    board = 48
    body_loofah = 49
    boiler = 50
    book = 51
    books = 52
    bookshelf = 53
    boots = 54
    bottle = 55
    bottles = 56
    bowl = 57
    box = 58
    boxes = 59
    bread = 60
    breadboard = 61
    brochure = 62
    brush = 63
    bucket = 64
    buggy = 65
    bulletin_board = 66
    cabinet = 67
    cable = 68
    cable_rack = 69
    calendar = 70
    can = 71
    candle = 72
    candles = 73
    candlestick = 74
    canopy = 75
    cap = 76
    carpet = 77
    carriage = 78
    cart = 79
    case = 80
    ceiling = 81
    ceiling_other_room = 82
    ceiling_light = 83
    chair = 84
    chairs = 85
    chandelier = 86
    changing_table = 87
    chest = 88
    child_chair = 89
    child_clothes = 90
    childrens_table = 91
    cleaning_agent = 92
    cleaning_brush = 93
    cleanser = 94
    clock = 95
    closet = 96
    closet_door = 97
    cloth = 98
    clothes = 99
    clothes_dryer = 100
    clothes_rack = 101
    clutter = 102
    coat = 103
    coffee = 104
    coffee_machine = 105
    coffee_maker = 106
    coffee_table = 107
    column = 108
    commode = 109
    computer = 110
    computer_desk = 111
    console = 112
    container = 113
    cooking_pot = 114
    corner_bench = 115
    cosmetics_kit = 116
    couch = 117
    couch_table = 118
    counter = 119
    cover = 120
    cradle = 121
    crate = 122
    crib = 123
    cube = 124
    cup = 125
    cupboard = 126
    cups = 127
    curtain = 128
    curtain_rail = 129
    cushion = 130
    cushions_stack = 131
    cut_board = 132
    cutting_board = 133
    cycling_trainer = 134
    darts = 135
    decoration = 136
    desk = 137
    desk_chair = 138
    device = 139
    diapers = 140
    dining_chair = 141
    dining_set = 142
    dining_table = 143
    discs = 144
    dish = 145
    dish_dryer = 146
    dishdrainer = 147
    dishes = 148
    dishwasher = 149
    dispenser = 150
    documents = 151
    dog = 152
    doll = 153
    door = 154
    door_other_room = 155
    door_mat = 156
    doorframe = 157
    doorframe_other_room = 158
    drain_pipe = 159
    drawer = 160
    drawers = 161
    drawers_rack = 162
    dress = 163
    dresser = 164
    dressing_table = 165
    drinks = 166
    drum = 167
    drying_machine = 168
    drying_rack = 169
    dumbbells = 170
    elevator = 171
    elliptical_trainer = 172
    exhaust_hood = 173
    exit_sign = 174
    extractor_fan = 175
    fabric = 176
    fan = 177
    fence = 178
    festoon = 179
    figure = 180
    file_cabinet = 181
    fire_extinguisher = 182
    fireplace = 183
    firewood_box = 184
    flag = 185
    flipchart = 186
    floor = 187
    floor_other_room = 188
    floor_lamp = 189
    floor_mat = 190
    flower = 191
    flowers = 192
    flush = 193
    folded_beach_chairs = 194
    folder = 195
    folding_chair = 196
    food = 197
    foosball_table = 198
    footstool = 199
    frame = 200
    fridge = 201
    fruit = 202
    fruit_plate = 203
    fruits = 204
    furniture = 205
    garbage = 206
    garbage_bin = 207
    garden_umbrella = 208
    generator = 209
    glass = 210
    glass_wall = 211
    grass = 212
    guitar = 213
    gymnastic_ball = 214
    hair_dryer = 215
    hand_brush = 216
    hand_dryer = 217
    hand_washer = 218
    handbag = 219
    handhold = 220
    handle = 221
    handrail = 222
    hanger = 223
    hangers = 224
    hanging_cabinet = 225
    headboard = 226
    heater = 227
    helmet = 228
    hood = 229
    humidifier = 230
    hygiene_products = 231
    instrument = 232
    iron = 233
    ironing_board = 234
    item = 235
    items = 236
    jacket = 237
    jalousie = 238
    jar = 239
    jug = 240
    juicer = 241
    kettle = 242
    keyboard = 243
    kids_bicycle = 244
    kids_chair = 245
    kids_rocking_chair = 246
    kids_stool = 247
    kids_table = 248
    kitchen_appliance = 249
    kitchen_cabinet = 250
    kitchen_counter = 251
    kitchen_hood = 252
    kitchen_item = 253
    kitchen_object = 254
    kitchen_playset = 255
    kitchen_rack = 256
    kitchen_sink = 257
    kitchen_sofa = 258
    kitchen_towel = 259
    knife_box = 260
    ladder = 261
    lamp = 262
    laptop = 263
    laundry_basket = 264
    letter = 265
    light = 266
    linen = 267
    locker = 268
    lockers = 269
    loft_bed = 270
    lounger = 271
    luggage = 272
    machine = 273
    magazine = 274
    magazine_files = 275
    magazine_rack = 276
    magazine_stand = 277
    mandarins = 278
    mannequin = 279
    mask = 280
    mattress = 281
    medical_device = 282
    menu = 283
    meter = 284
    microwave = 285
    milk = 286
    mirror = 287
    monitor = 288
    mop = 289
    multicooker = 290
    napkins = 291
    newspaper = 292
    newspaper_rack = 293
    nightstand = 294
    notebook = 295
    notebooks = 296
    object = 297
    objects = 298
    office_chair = 299
    office_table = 300
    organizer = 301
    ottoman = 302
    oven = 303
    oven_glove = 304
    pack = 305
    package = 306
    packs = 307
    painting = 308
    pan = 309
    paper = 310
    paper_cutter = 311
    paper_holder = 312
    paper_sign = 313
    paper_stack = 314
    paper_towel = 315
    paper_towel_dispenser = 316
    papers = 317
    partition = 318
    pavement = 319
    pc = 320
    pepper = 321
    pet_bed = 322
    photo_frame = 323
    photos = 324
    piano = 325
    picture = 326
    pictures = 327
    pile = 328
    pile_of_books = 329
    pile_of_bottles = 330
    pile_of_candles = 331
    pile_of_folders = 332
    pile_of_papers = 333
    pile_of_pillows = 334
    pile_of_wires = 335
    pillar = 336
    pillow = 337
    pin_board_wall = 338
    pipe = 339
    plank = 340
    plant = 341
    planter = 342
    plants = 343
    plate = 344
    plates = 345
    platform = 346
    player = 347
    pocket = 348
    podest = 349
    pooh = 350
    poster = 351
    pot = 352
    price_tag = 353
    printer = 354
    projector = 355
    puf = 356
    puppet = 357
    rack = 358
    radiator = 359
    radio = 360
    rag = 361
    rail = 362
    railing = 363
    ramp = 364
    recycle_bin = 365
    refrigerator = 366
    rocking_chair = 367
    roll = 368
    rolled_carpet = 369
    rolling_cart = 370
    rolling_pin = 371
    roof = 372
    round_table = 373
    rowing_machine = 374
    rubbish_bin = 375
    rug = 376
    sack = 377
    salad = 378
    salt = 379
    sauce_boat = 380
    scale = 381
    scarf = 382
    screen = 383
    seat = 384
    seat_pad = 385
    sewing_machine = 386
    shades = 387
    shampoo = 388
    sheets = 389
    shelf = 390
    shelf_clutter = 391
    shelf_of_caps = 392
    shelf_unit = 393
    shelves = 394
    shirt = 395
    shoe = 396
    shoe_box = 397
    shoe_commode = 398
    shoe_rack = 399
    shoe_shelf = 400
    shoes = 401
    showcase = 402
    shower = 403
    shower_curtain = 404
    shower_door = 405
    shower_floor = 406
    shower_gel = 407
    shower_wall = 408
    side_table = 409
    sideboard = 410
    sidecouch = 411
    sidetable = 412
    sign = 413
    sink = 414
    sink_counter = 415
    slanted_wall = 416
    snowboard = 417
    soap = 418
    soap_dish = 419
    soap_dispenser = 420
    socket = 421
    sofa = 422
    sofa_chair = 423
    sofa_couch = 424
    speaker = 425
    spice = 426
    spices = 427
    sponge = 428
    spots = 429
    squeezer = 430
    stair = 431
    stairs = 432
    stand = 433
    star = 434
    statue = 435
    statuette = 436
    stepladder = 437
    stereo = 438
    stereo_equipment = 439
    stick = 440
    stool = 441
    storage = 442
    storage_bin = 443
    storage_box = 444
    storage_container = 445
    storage_unit = 446
    stove = 447
    stroller = 448
    stuffed_animal = 449
    sugar_packs = 450
    suitcase = 451
    switch = 452
    t_shirt = 453
    table = 454
    table_lamp = 455
    table_soccer = 456
    tablet = 457
    teapot = 458
    teddy_bear = 459
    telephone = 460
    tennis_raquet = 461
    tent = 462
    things = 463
    tile = 464
    tire = 465
    tissue_pack = 466
    toaster = 467
    toilet = 468
    toilet_brush = 469
    toilet_paper = 470
    toilet_paper_dispenser = 471
    toilet_paper_holder = 472
    toiletry = 473
    tool_wall = 474
    towel = 475
    towel_basket = 476
    towels = 477
    toy = 478
    toy_house = 479
    trash_bin = 480
    trash_can = 481
    trashcan = 482
    tray = 483
    treadmill = 484
    tree = 485
    tree_decoration = 486
    tube = 487
    tv = 488
    tv_stand = 489
    tv_table = 490
    typewriter = 491
    ukulele = 492
    umbrella = 493
    upholstered_wall = 494
    urinal = 495
    utensils = 496
    vacuum = 497
    vacuum_cleaner = 498
    vase = 499
    ventilation = 500
    ventilator = 501
    wall = 502
    wall_other_room = 503
    wall_frame = 504
    wall_plants = 505
    wall_rack = 506
    wardrobe = 507
    wardrobe_door = 508
    washbasin = 509
    washing_basket = 510
    washing_machine = 511
    washing_powder = 512
    water = 513
    water_heater = 514
    watering_can = 515
    weights = 516
    weigths = 517
    whiteboard = 518
    window = 519
    window_board = 520
    window_clutter = 521
    window_frame = 522
    windows = 523
    windowsill = 524
    wood = 525
    wood_box = 526
    xbox = 527


object_classifications = [

    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=0, label='air conditioner'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=40, rio=1, label='apron'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=2, label='aquarium'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=3, label='armchair'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=3, rio=4, label='armoire'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=40, rio=5, label='armor'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=6, label='audio system'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=7, label='baby bed'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=8, label='baby changing table'),
    ObjectClassification(eigen13=6, rio27=7, nyu40=39, rio=9, label='baby changing unit'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=10, label='baby gym'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=11, label='baby seat'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=12, label='baby toys'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=13, label='backpack'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=37, rio=14, label='bag'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=15, label='balcony'),
    ObjectClassification(eigen13=0, rio27=8, nyu40=8, rio=16, label='balcony door'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=17, label='ball'),
    ObjectClassification(eigen13=0, rio27=10, nyu40=38, rio=18, label='bar'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=19, label='bar stool'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=20, label='barrel'),
    ObjectClassification(eigen13=7, rio27=5, nyu40=40, rio=21, label='barstool'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=22, label='basin'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=40, rio=23, label='basket'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=3, rio=24, label='bath cabinet'),
    ObjectClassification(eigen13=6, rio27=10, nyu40=12, rio=25, label='bath counter'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=15, rio=26, label='bath rack'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=27, label='bath robe'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=28, label='bathrobe'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=29, label='bathroom items'),
    ObjectClassification(eigen13=6, rio27=25, nyu40=36, rio=30, label='bathtub'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=31, label='bbq'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=32, label='beam'),
    ObjectClassification(eigen13=6, rio27=26, nyu40=39, rio=33, label='bean bag'),
    ObjectClassification(eigen13=6, rio27=26, nyu40=39, rio=34, label='beanbag'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=35, label='beautician'),
    ObjectClassification(eigen13=1, rio27=4, nyu40=4, rio=36, label='bed'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=37, label='bed table'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=38, label='bedside table'),
    ObjectClassification(eigen13=6, rio27=5, nyu40=39, rio=39, label='bench'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=29, rio=40, label='beverage crate'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=41, label='bicycle'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=42, label='bidet'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=43, label='bike'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=44, label='bin'),
    ObjectClassification(eigen13=0, rio27=1, nyu40=38, rio=45, label='blackboard'),
    ObjectClassification(eigen13=7, rio27=27, nyu40=40, rio=46, label='blanket'),
    ObjectClassification(eigen13=7, rio27=9, nyu40=13, rio=47, label='blinds'),
    ObjectClassification(eigen13=0, rio27=26, nyu40=38, rio=48, label='board'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=49, label='body loofah'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=50, label='boiler'),
    ObjectClassification(eigen13=2, rio27=0, nyu40=23, rio=51, label='book'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=52, label='books'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=10, rio=53, label='bookshelf'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=54, label='boots'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=55, label='bottle'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=56, label='bottles'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=57, label='bowl'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=29, rio=58, label='box'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=29, rio=59, label='boxes'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=60, label='bread'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=61, label='breadboard'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=62, label='brochure'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=63, label='brush'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=40, rio=64, label='bucket'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=65, label='buggy'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=66, label='bulletin board'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=3, rio=67, label='cabinet'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=68, label='cable'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=15, rio=69, label='cable rack'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=70, label='calendar'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=71, label='can'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=72, label='candle'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=73, label='candles'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=74, label='candlestick'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=75, label='canopy'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=76, label='cap'),
    ObjectClassification(eigen13=7, rio27=2, nyu40=40, rio=77, label='carpet'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=78, label='carriage'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=79, label='cart'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=80, label='case'),
    ObjectClassification(eigen13=3, rio27=15, nyu40=22, rio=81, label='ceiling'),
    ObjectClassification(eigen13=3, rio27=15, nyu40=22, rio=82, label='ceiling /other room'),
    ObjectClassification(eigen13=7, rio27=24, nyu40=35, rio=83, label='ceiling light'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=84, label='chair'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=85, label='chairs'),
    ObjectClassification(eigen13=0, rio27=24, nyu40=35, rio=86, label='chandelier'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=87, label='changing table'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=88, label='chest'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=89, label='child chair'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=21, rio=90, label='child clothes'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=91, label="children's table"),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=92, label='cleaning agent'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=93, label='cleaning brush'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=94, label='cleanser'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=95, label='clock'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=39, rio=96, label='closet'),
    ObjectClassification(eigen13=0, rio27=8, nyu40=8, rio=97, label='closet door'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=40, rio=98, label='cloth'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=21, rio=99, label='clothes'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=100, label='clothes dryer'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=15, rio=101, label='clothes rack'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=102, label='clutter'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=40, rio=103, label='coat'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=104, label='coffee'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=105, label='coffee machine'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=106, label='coffee maker'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=107, label='coffee table'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=108, label='column'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=3, rio=109, label='commode'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=110, label='computer'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=14, rio=111, label='computer desk'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=112, label='console'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=40, rio=113, label='container'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=114, label='cooking pot'),
    ObjectClassification(eigen13=6, rio27=5, nyu40=39, rio=115, label='corner bench'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=116, label='cosmetics kit'),
    ObjectClassification(eigen13=9, rio27=6, nyu40=6, rio=117, label='couch'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=118, label='couch table'),
    ObjectClassification(eigen13=6, rio27=10, nyu40=12, rio=119, label='counter'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=120, label='cover'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=121, label='cradle'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=122, label='crate'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=123, label='crib'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=124, label='cube'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=125, label='cup'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=3, rio=126, label='cupboard'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=127, label='cups'),
    ObjectClassification(eigen13=7, rio27=12, nyu40=16, rio=128, label='curtain'),
    ObjectClassification(eigen13=7, rio27=12, nyu40=16, rio=129, label='curtain rail'),
    ObjectClassification(eigen13=7, rio27=13, nyu40=18, rio=130, label='cushion'),
    ObjectClassification(eigen13=7, rio27=13, nyu40=18, rio=131, label='cushions stack'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=132, label='cut board'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=133, label='cutting board'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=134, label='cycling trainer'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=135, label='darts'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=136, label='decoration'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=14, rio=137, label='desk'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=138, label='desk chair'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=139, label='device'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=140, label='diapers'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=141, label='dining chair'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=142, label='dining set'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=143, label='dining table'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=144, label='discs'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=145, label='dish'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=146, label='dish dryer'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=147, label='dishdrainer'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=148, label='dishes'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=149, label='dishwasher'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=150, label='dispenser'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=26, rio=151, label='documents'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=152, label='dog'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=153, label='doll'),
    ObjectClassification(eigen13=0, rio27=8, nyu40=8, rio=154, label='door'),
    ObjectClassification(eigen13=0, rio27=8, nyu40=8, rio=155, label='door /other room'),
    ObjectClassification(eigen13=7, rio27=2, nyu40=20, rio=156, label='door mat'),
    ObjectClassification(eigen13=0, rio27=8, nyu40=38, rio=157, label='doorframe'),
    ObjectClassification(eigen13=0, rio27=8, nyu40=38, rio=158, label='doorframe /other room'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=159, label='drain pipe'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=39, rio=160, label='drawer'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=17, rio=161, label='drawers'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=162, label='drawers rack'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=40, rio=163, label='dress'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=17, rio=164, label='dresser'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=165, label='dressing table'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=166, label='drinks'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=167, label='drum'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=168, label='drying machine'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=169, label='drying rack'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=170, label='dumbbells'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=171, label='elevator'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=172, label='elliptical trainer'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=173, label='exhaust hood'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=174, label='exit sign'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=175, label='extractor fan'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=40, rio=176, label='fabric'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=177, label='fan'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=178, label='fence'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=179, label='festoon'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=180, label='figure'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=3, rio=181, label='file cabinet'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=182, label='fire extinguisher'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=183, label='fireplace'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=29, rio=184, label='firewood box'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=185, label='flag'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=186, label='flipchart'),
    ObjectClassification(eigen13=5, rio27=2, nyu40=2, rio=187, label='floor'),
    ObjectClassification(eigen13=5, rio27=2, nyu40=2, rio=188, label='floor /other room'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=35, rio=189, label='floor lamp'),
    ObjectClassification(eigen13=7, rio27=2, nyu40=20, rio=190, label='floor mat'),
    ObjectClassification(eigen13=7, rio27=19, nyu40=40, rio=191, label='flower'),
    ObjectClassification(eigen13=7, rio27=19, nyu40=40, rio=192, label='flowers'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=193, label='flush'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=194, label='folded beach chairs'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=195, label='folder'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=196, label='folding chair'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=197, label='food'),
    ObjectClassification(eigen13=7, rio27=7, nyu40=40, rio=198, label='foosball table'),
    ObjectClassification(eigen13=6, rio27=5, nyu40=39, rio=199, label='footstool'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=200, label='frame'),
    ObjectClassification(eigen13=6, rio27=16, nyu40=24, rio=201, label='fridge'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=202, label='fruit'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=203, label='fruit plate'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=204, label='fruits'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=205, label='furniture'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=206, label='garbage'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=207, label='garbage bin'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=208, label='garden umbrella'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=209, label='generator'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=210, label='glass'),
    ObjectClassification(eigen13=12, rio27=1, nyu40=1, rio=211, label='glass wall'),
    ObjectClassification(eigen13=0, rio27=19, nyu40=38, rio=212, label='grass'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=213, label='guitar'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=214, label='gymnastic ball'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=215, label='hair dryer'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=216, label='hand brush'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=217, label='hand dryer'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=218, label='hand washer'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=219, label='handbag'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=220, label='handhold'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=221, label='handle'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=222, label='handrail'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=223, label='hanger'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=224, label='hangers'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=3, rio=225, label='hanging cabinet'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=226, label='headboard'),
    ObjectClassification(eigen13=6, rio27=26, nyu40=39, rio=227, label='heater'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=228, label='helmet'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=229, label='hood'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=230, label='humidifier'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=231, label='hygiene products'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=232, label='instrument'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=233, label='iron'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=234, label='ironing board'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=235, label='item'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=236, label='items'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=40, rio=237, label='jacket'),
    ObjectClassification(eigen13=7, rio27=9, nyu40=13, rio=238, label='jalousie'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=239, label='jar'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=240, label='jug'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=241, label='juicer'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=242, label='kettle'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=243, label='keyboard'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=244, label='kids bicycle'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=245, label='kids chair'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=246, label='kids rocking chair'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=247, label='kids stool'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=248, label='kids table'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=249, label='kitchen appliance'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=3, rio=250, label='kitchen cabinet'),
    ObjectClassification(eigen13=6, rio27=10, nyu40=12, rio=251, label='kitchen counter'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=252, label='kitchen hood'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=253, label='kitchen item'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=254, label='kitchen object'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=255, label='kitchen playset'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=15, rio=256, label='kitchen rack'),
    ObjectClassification(eigen13=6, rio27=23, nyu40=34, rio=257, label='kitchen sink'),
    ObjectClassification(eigen13=9, rio27=6, nyu40=6, rio=258, label='kitchen sofa'),
    ObjectClassification(eigen13=7, rio27=18, nyu40=27, rio=259, label='kitchen towel'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=29, rio=260, label='knife box'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=261, label='ladder'),
    ObjectClassification(eigen13=7, rio27=24, nyu40=35, rio=262, label='lamp'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=263, label='laptop'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=40, rio=264, label='laundry basket'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=26, rio=265, label='letter'),
    ObjectClassification(eigen13=0, rio27=24, nyu40=38, rio=266, label='light'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=21, rio=267, label='linen'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=3, rio=268, label='locker'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=269, label='lockers'),
    ObjectClassification(eigen13=1, rio27=4, nyu40=4, rio=270, label='loft bed'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=271, label='lounger'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=272, label='luggage'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=273, label='machine'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=274, label='magazine'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=275, label='magazine files'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=15, rio=276, label='magazine rack'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=277, label='magazine stand'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=278, label='mandarins'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=279, label='mannequin'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=280, label='mask'),
    ObjectClassification(eigen13=7, rio27=4, nyu40=40, rio=281, label='mattress'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=282, label='medical device'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=283, label='menu'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=284, label='meter'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=285, label='microwave'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=286, label='milk'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=19, rio=287, label='mirror'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=288, label='monitor'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=289, label='mop'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=290, label='multicooker'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=291, label='napkins'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=26, rio=292, label='newspaper'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=15, rio=293, label='newspaper rack'),
    ObjectClassification(eigen13=6, rio27=21, nyu40=32, rio=294, label='nightstand'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=295, label='notebook'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=296, label='notebooks'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=297, label='object'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=298, label='objects'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=299, label='office chair'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=300, label='office table'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=301, label='organizer'),
    ObjectClassification(eigen13=6, rio27=6, nyu40=39, rio=302, label='ottoman'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=303, label='oven'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=40, rio=304, label='oven glove'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=305, label='pack'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=306, label='package'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=307, label='packs'),
    ObjectClassification(eigen13=8, rio27=1, nyu40=11, rio=308, label='painting'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=309, label='pan'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=26, rio=310, label='paper'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=311, label='paper cutter'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=312, label='paper holder'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=26, rio=313, label='paper sign'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=26, rio=314, label='paper stack'),
    ObjectClassification(eigen13=7, rio27=18, nyu40=40, rio=315, label='paper towel'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=316, label='paper towel dispenser'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=26, rio=317, label='papers'),
    ObjectClassification(eigen13=12, rio27=1, nyu40=1, rio=318, label='partition'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=319, label='pavement'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=320, label='pc'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=321, label='pepper'),
    ObjectClassification(eigen13=7, rio27=13, nyu40=40, rio=322, label='pet bed'),
    ObjectClassification(eigen13=8, rio27=0, nyu40=11, rio=323, label='photo frame'),
    ObjectClassification(eigen13=8, rio27=0, nyu40=11, rio=324, label='photos'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=325, label='piano'),
    ObjectClassification(eigen13=8, rio27=0, nyu40=11, rio=326, label='picture'),
    ObjectClassification(eigen13=8, rio27=26, nyu40=11, rio=327, label='pictures'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=328, label='pile'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=329, label='pile of books'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=330, label='pile of bottles'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=331, label='pile of candles'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=26, rio=332, label='pile of folders'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=26, rio=333, label='pile of papers'),
    ObjectClassification(eigen13=7, rio27=13, nyu40=18, rio=334, label='pile of pillows'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=335, label='pile of wires'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=336, label='pillar'),
    ObjectClassification(eigen13=7, rio27=13, nyu40=18, rio=337, label='pillow'),
    ObjectClassification(eigen13=7, rio27=1, nyu40=40, rio=338, label='pin board wall'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=339, label='pipe'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=340, label='plank'),
    ObjectClassification(eigen13=7, rio27=19, nyu40=40, rio=341, label='plant'),
    ObjectClassification(eigen13=7, rio27=19, nyu40=40, rio=342, label='planter'),
    ObjectClassification(eigen13=7, rio27=19, nyu40=40, rio=343, label='plants'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=344, label='plate'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=345, label='plates'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=346, label='platform'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=347, label='player'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=348, label='pocket'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=349, label='podest'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=350, label='pooh'),
    ObjectClassification(eigen13=8, rio27=1, nyu40=11, rio=351, label='poster'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=352, label='pot'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=353, label='price tag'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=354, label='printer'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=355, label='projector'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=356, label='puf'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=357, label='puppet'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=358, label='rack'),
    ObjectClassification(eigen13=6, rio27=26, nyu40=39, rio=359, label='radiator'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=360, label='radio'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=27, rio=361, label='rag'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=362, label='rail'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=363, label='railing'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=364, label='ramp'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=365, label='recycle bin'),
    ObjectClassification(eigen13=6, rio27=16, nyu40=24, rio=366, label='refrigerator'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=367, label='rocking chair'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=368, label='roll'),
    ObjectClassification(eigen13=7, rio27=2, nyu40=20, rio=369, label='rolled carpet'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=370, label='rolling cart'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=371, label='rolling pin'),
    ObjectClassification(eigen13=3, rio27=15, nyu40=22, rio=372, label='roof'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=373, label='round table'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=374, label='rowing machine'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=375, label='rubbish bin'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=376, label='rug'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=37, rio=377, label='sack'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=378, label='salad'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=379, label='salt'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=380, label='sauce boat'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=381, label='scale'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=40, rio=382, label='scarf'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=383, label='screen'),
    ObjectClassification(eigen13=6, rio27=5, nyu40=39, rio=384, label='seat'),
    ObjectClassification(eigen13=7, rio27=13, nyu40=18, rio=385, label='seat pad'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=386, label='sewing machine'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=387, label='shades'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=388, label='shampoo'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=389, label='sheets'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=15, rio=390, label='shelf'),
    ObjectClassification(eigen13=7, rio27=11, nyu40=40, rio=391, label='shelf clutter'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=15, rio=392, label='shelf of caps'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=15, rio=393, label='shelf unit'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=15, rio=394, label='shelves'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=40, rio=395, label='shirt'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=396, label='shoe'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=29, rio=397, label='shoe box'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=17, rio=398, label='shoe commode'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=15, rio=399, label='shoe rack'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=15, rio=400, label='shoe shelf'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=401, label='shoes'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=3, rio=402, label='showcase'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=403, label='shower'),
    ObjectClassification(eigen13=7, rio27=12, nyu40=28, rio=404, label='shower curtain'),
    ObjectClassification(eigen13=0, rio27=8, nyu40=8, rio=405, label='shower door'),
    ObjectClassification(eigen13=5, rio27=2, nyu40=2, rio=406, label='shower floor'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=407, label='shower gel'),
    ObjectClassification(eigen13=12, rio27=1, nyu40=1, rio=408, label='shower wall'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=409, label='side table'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=12, rio=410, label='sideboard'),
    ObjectClassification(eigen13=9, rio27=6, nyu40=6, rio=411, label='sidecouch'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=412, label='sidetable'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=413, label='sign'),
    ObjectClassification(eigen13=6, rio27=23, nyu40=34, rio=414, label='sink'),
    ObjectClassification(eigen13=6, rio27=10, nyu40=12, rio=415, label='sink counter'),
    ObjectClassification(eigen13=12, rio27=1, nyu40=1, rio=416, label='slanted wall'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=417, label='snowboard'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=418, label='soap'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=419, label='soap dish'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=420, label='soap dispenser'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=421, label='socket'),
    ObjectClassification(eigen13=9, rio27=6, nyu40=6, rio=422, label='sofa'),
    ObjectClassification(eigen13=4, rio27=5, nyu40=5, rio=423, label='sofa chair'),
    ObjectClassification(eigen13=9, rio27=6, nyu40=6, rio=424, label='sofa couch'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=425, label='speaker'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=426, label='spice'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=427, label='spices'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=428, label='sponge'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=429, label='spots'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=430, label='squeezer'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=431, label='stair'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=432, label='stairs'),
    ObjectClassification(eigen13=6, rio27=7, nyu40=39, rio=433, label='stand'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=434, label='star'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=435, label='statue'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=436, label='statuette'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=437, label='stepladder'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=438, label='stereo'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=439, label='stereo equipment'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=440, label='stick'),
    ObjectClassification(eigen13=7, rio27=5, nyu40=40, rio=441, label='stool'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=442, label='storage'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=443, label='storage bin'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=29, rio=444, label='storage box'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=29, rio=445, label='storage container'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=39, rio=446, label='storage unit'),
    ObjectClassification(eigen13=0, rio27=0, nyu40=38, rio=447, label='stove'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=448, label='stroller'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=449, label='stuffed animal'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=450, label='sugar packs'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=451, label='suitcase'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=452, label='switch'),
    ObjectClassification(eigen13=7, rio27=14, nyu40=40, rio=453, label='t-shirt'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=454, label='table'),
    ObjectClassification(eigen13=7, rio27=24, nyu40=35, rio=455, label='table lamp'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=456, label='table soccer'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=457, label='tablet'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=458, label='teapot'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=459, label='teddy bear'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=460, label='telephone'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=461, label='tennis raquet'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=462, label='tent'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=463, label='things'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=464, label='tile'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=465, label='tire'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=466, label='tissue pack'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=467, label='toaster'),
    ObjectClassification(eigen13=6, rio27=22, nyu40=33, rio=468, label='toilet'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=469, label='toilet brush'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=470, label='toilet paper'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=471, label='toilet paper dispenser'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=472, label='toilet paper holder'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=473, label='toiletry'),
    ObjectClassification(eigen13=7, rio27=1, nyu40=40, rio=474, label='tool wall'),
    ObjectClassification(eigen13=7, rio27=18, nyu40=27, rio=475, label='towel'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=476, label='towel basket'),
    ObjectClassification(eigen13=7, rio27=18, nyu40=27, rio=477, label='towels'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=478, label='toy'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=479, label='toy house'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=480, label='trash bin'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=481, label='trash can'),
    ObjectClassification(eigen13=6, rio27=26, nyu40=39, rio=482, label='trashcan'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=483, label='tray'),
    ObjectClassification(eigen13=6, rio27=0, nyu40=39, rio=484, label='treadmill'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=485, label='tree'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=486, label='tree decoration'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=487, label='tube'),
    ObjectClassification(eigen13=11, rio27=17, nyu40=25, rio=488, label='tv'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=39, rio=489, label='tv stand'),
    ObjectClassification(eigen13=10, rio27=7, nyu40=7, rio=490, label='tv table'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=491, label='typewriter'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=492, label='ukulele'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=493, label='umbrella'),
    ObjectClassification(eigen13=12, rio27=1, nyu40=1, rio=494, label='upholstered wall'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=495, label='urinal'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=496, label='utensils'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=497, label='vacuum'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=498, label='vacuum cleaner'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=499, label='vase'),
    ObjectClassification(eigen13=0, rio27=26, nyu40=38, rio=500, label='ventilation'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=501, label='ventilator'),
    ObjectClassification(eigen13=12, rio27=1, nyu40=1, rio=502, label='wall'),
    ObjectClassification(eigen13=12, rio27=1, nyu40=1, rio=503, label='wall /other room'),
    ObjectClassification(eigen13=8, rio27=1, nyu40=11, rio=504, label='wall frame'),
    ObjectClassification(eigen13=7, rio27=19, nyu40=40, rio=505, label='wall plants'),
    ObjectClassification(eigen13=6, rio27=11, nyu40=15, rio=506, label='wall rack'),
    ObjectClassification(eigen13=6, rio27=3, nyu40=39, rio=507, label='wardrobe'),
    ObjectClassification(eigen13=0, rio27=8, nyu40=8, rio=508, label='wardrobe door'),
    ObjectClassification(eigen13=6, rio27=23, nyu40=34, rio=509, label='washbasin'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=510, label='washing basket'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=511, label='washing machine'),
    ObjectClassification(eigen13=7, rio27=26, nyu40=40, rio=512, label='washing powder'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=513, label='water'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=514, label='water heater'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=515, label='watering can'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=516, label='weights'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=517, label='weigths'),
    ObjectClassification(eigen13=7, rio27=1, nyu40=30, rio=518, label='whiteboard'),
    ObjectClassification(eigen13=13, rio27=9, nyu40=9, rio=519, label='window'),
    ObjectClassification(eigen13=0, rio27=9, nyu40=38, rio=520, label='window board'),
    ObjectClassification(eigen13=7, rio27=9, nyu40=40, rio=521, label='window clutter'),
    ObjectClassification(eigen13=0, rio27=9, nyu40=38, rio=522, label='window frame'),
    ObjectClassification(eigen13=13, rio27=9, nyu40=9, rio=523, label='windows'),
    ObjectClassification(eigen13=0, rio27=9, nyu40=38, rio=524, label='windowsill'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=525, label='wood'),
    ObjectClassification(eigen13=7, rio27=20, nyu40=29, rio=526, label='wood box'),
    ObjectClassification(eigen13=7, rio27=0, nyu40=40, rio=527, label='xbox'),
]
