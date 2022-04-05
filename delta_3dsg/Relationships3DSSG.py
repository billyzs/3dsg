from enum import IntEnum, unique

@unique
class Relationships3DSSG(IntEnum):
    def __str__(self):
        return super().__str__().replace("Objects","")
    none = 0
    supported_by = 1
    left = 2
    right = 3
    front = 4
    behind = 5
    close_by = 6
    inside = 7
    bigger_than = 8
    smaller_than = 9
    higher_than = 10
    lower_than = 11
    same_symmetry_as = 12
    same_as = 13
    attached_to = 14
    standing_on = 15
    lying_on = 16
    hanging_on = 17
    connected_to = 18
    leaning_against = 19
    part_of = 20
    belonging_to = 21
    build_in = 22
    standing_in = 23
    cover = 24
    lying_in = 25
    hanging_in = 26
    same_color = 27
    same_material = 28
    same_texture = 29
    same_shape = 30
    same_state = 31
    same_object_type = 32
    messier_than = 33
    cleaner_than = 34
    fuller_than = 35
    more_closed = 36
    more_open = 37
    brighter_than = 38
    darker_than = 39
    more_comfortable_than = 40
    num_relationships = 41
