format:
	columns:
		data (Normalized) -> [position, road type, speed limit, mass, elevation change, previous orientation, length, direction angle]
		label (10ml,s)-> [fuel consumption, time]
		segment_id : id of the segment
		length_of the path
 		position in the path


data:
	position: Relative position in trip
	road type
	speed limit
	mass
	elevation change
	previous orientation: Turning angle from the previous segment (0 in default, right turn > 0, left turn < 0)
	length
	direction angle: Direction angle of the segment based on east direction (north>0)