from frame_processing import *
from shapely.geometry import Point, MultiPoint

class Obstacle():
    def __init__(self,obj_id,obj_height,map_points,size,scale,displacement,trophy_prob =None, number_pixel_points=0):
        self.id = obj_id
        self.occupancy_map_points = map_points[:]
        self.height = obj_height
        self.is_goal = False
        self.current_frame_id = None
        self.is_container = 1 #initial assumption that anything can be a container
        self.is_opened = False
        self.centre_x = None
        self.centre_y = None
        self.centre_z = None
        self.bounding_box = None
        self.calculate_bounding_box(size,scale, displacement)
        self.calculate_centre()
        self.parent_id = -1
        self.is_picked_and_not_trophy = False
        self.number_pixel_points = [number_pixel_points]
        if trophy_prob == None :
            self.trophy_prob = 0
            self.trophy_prob_per_frame = []
        else :
            self.trophy_prob = trophy_prob
            self.trophy_prob_per_frame = [trophy_prob]
        #self.agent_pos_per_frame = [agent_pos]

    def expand_obstacle(self, current_scene_map_points,size,scale,displacement):
        obstacle_len = len(self.occupancy_map_points)
        self.occupancy_map_points = self.union(self.occupancy_map_points, current_scene_map_points)
        obstacle_len_updated = len(self.occupancy_map_points)
        if obstacle_len_updated > obstacle_len :
            self.calculate_bounding_box(size,scale,displacement)
            self.calculate_centre()

    def calculate_bounding_box(self,size, scale,displacement):   
        obj_occ_map = get_occupancy_from_points( self.occupancy_map_points,size)   
        self.bounding_box = polygon_simplify(occupancy_to_polygons(obj_occ_map,scale,displacement ))

    def calculate_centre(self):
        exterior_coords = self.get_convex_polygon_coords()
        self.centre_x = np.mean(np.array(exterior_coords[0],dtype=object))
        self.centre_z = np.mean(np.array(exterior_coords[1],dtype=object))
        self.centre_y = self.height / 2.0

    def get_convex_polygon_coords(self):
        if self.bounding_box.geom_type == "MultiPolygon":
            bd_point = set()
            for polygon in self.bounding_box :
                x_list, z_list = polygon.exterior.coords.xy
                for x,z in zip(x_list,z_list):
                    if (x, z) not in bd_point:
                        bd_point.add((x, z))

            poly = MultiPoint(sorted(bd_point)).convex_hull
            self.convex_bounding_box = poly.simplify(0.0)#MultiPoint(sorted(bd_point)).convex_hull
            exterior_coords = self.convex_bounding_box.exterior.coords.xy
        else: 
            exterior_coords = self.bounding_box.exterior.coords.xy

        return exterior_coords

    def calculate_trophy_prob(self):
        number_top_elem = 3
        sorted_trophy_prob = sorted(self.trophy_prob_per_frame,reverse=True)
        if len(sorted_trophy_prob) == 0 :
            self.trophy_prob = 0
            return
    
        if len(sorted_trophy_prob) > number_top_elem :
            self.trophy_prob = sum(sorted_trophy_prob[:number_top_elem])/number_top_elem
        else :
            self.trophy_prob = sum(sorted_trophy_prob)/len(sorted_trophy_prob)
            
        
    def get_occupancy_map_points(self):
        return self.occupancy_map_points

    def get_centre(self):
        return (self.centre_x,self.centre_y,self.centre_z)

    def get_bounding_box(self): 
        return self.bounding_box

    def get_height(self):
        return self.height

    def is_possible_trophy (self) :
        if self.is_picked_and_not_trophy :
            return False
        if self.height > 2.5 : #Need to change this to trophy height threshold
            return False
        if self.is_opened == True :
            return False
        return True

    def is_possible_container(self):
        if self.height < 0.1 :
            return False
        if self.height > 2.5 :#Need go change this to container height threshold
            return False
        if self.is_opened == True :
            return False
        return True
    
    def union(self, a, b):
        """ return the union of two lists """
        return np.array([x for x in set(tuple(x) for x in a) | set(tuple(x) for x in b)])
        #return list(set(a) | set(b))
