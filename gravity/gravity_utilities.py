def convert_l2_to_dict(metadata):
        step_output_dict = {
            "camera_field_of_view": 42.5,
            "camera_aspect_ratio": (600, 400),
            "structural_object_list": {
                "floor": {
                    "dimensions": metadata.floor.dims,
                    "position": metadata.floor.centroid,
                    "color": metadata.floor.color,
                    "shape": metadata.floor.kind
                }
            },
            "object_list": {}
        }

        if hasattr(metadata, "poles"):
            step_output_dict["structural_object_list"]["poles"] = []

            for pole in metadata.poles:
                step_output_dict["structural_object_list"]["poles"].append({
                    "dimensions": pole.dims,
                    "position": pole.centroid,
                    "color": pole.color,
                    "shape": pole.kind
                })
        
        if hasattr(metadata, "occluders"):
            step_output_dict["structural_object_list"]["occluders"] = []

            for occluder in metadata.occluders:
                step_output_dict["structural_object_list"]["occluders"].append({
                    "dimensions": occluder.dims,
                    "position": occluder.centroid,
                    "color": occluder.color,
                    "shape": occluder.kind
                })
        

        if hasattr(metadata, "targets"):
            step_output_dict["object_list"]["targets"] = []
            
            for target in metadata.targets:
                step_output_dict["object_list"]["targets"].append({
                    "dimensions": target.dims,
                    "position": {
                        "x": target.centroid[0],
                        "y": target.centroid[1],
                        "z": target.centroid[2]
                    },
                    "color": {
                            "r": target.color[0],
                            "g": target.color[1],
                            "b": target.color[2],
                    },
                    "shape": target.kind,
                    "mass": 10.0,
                    "pixel_center": target.centroid_px
                })

        return step_output_dict