"""
synthetic_city.py

Synthetic city generator used by all paper figures.

Author : Tushar
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import pickle
import numpy as np

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

SEED = 42
RNG = np.random.default_rng(SEED)

AREA_WIDTH = 1000
AREA_HEIGHT = 1000

CITY_NAME = "Synthetic Smart City"

TOTAL_NODES = 500

LOW_NODES = 350
MEDIUM_NODES = 100
HIGH_NODES = 50

CITY_FILE = Path("synthetic_city.pkl")

# =============================================================================
# DISTRICT CONFIGURATION
# =============================================================================

DISTRICT_CONFIG = {

    "Downtown": {

        "center": (500, 500),

        "radius": 170,

        "building_height": (25, 40),

        "building_density": 0.90,

    },

    "Commercial": {

        "center": (500, 760),

        "radius": 140,

        "building_height": (15, 28),

        "building_density": 0.75,

    },

    "Residential_A": {

        "center": (220, 500),

        "radius": 180,

        "building_height": (5, 15),

        "building_density": 0.55,

    },

    "Residential_B": {

        "center": (780, 500),

        "radius": 180,

        "building_height": (5, 15),

        "building_density": 0.55,

    },

    "Industrial": {

        "center": (780, 820),

        "radius": 150,

        "building_height": (20, 35),

        "building_density": 0.70,

    },

    "Hospital": {

        "center": (500, 180),

        "radius": 100,

        "building_height": (15, 25),

        "building_density": 0.65,

    },

    "PowerStation": {

        "center": (170, 820),

        "radius": 100,

        "building_height": (12, 20),

        "building_density": 0.60,

    }

}

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class Metadata:

    city_name: str

    width: int

    height: int

    seed: int


@dataclass
class District:

    id: int

    name: str

    center_x: float

    center_y: float

    radius: float

    building_height_min: float

    building_height_max: float

    building_density: float


@dataclass
class Road:

    id: int

    x1: float
    y1: float

    x2: float
    y2: float

    road_type: str


@dataclass
class Building:

    id: int

    district: str

    poi: str

    x: float
    y: float

    width: float
    depth: float

    height: float


@dataclass
class PointOfInterest:

    id: int

    name: str

    district: str

    x: float
    y: float

    category: str


@dataclass
class IoTNode:

    id: int

    district: str

    building_id: int

    x: float
    y: float
    z: float

    priority: str

    demand: float

    required_rate: float


@dataclass
class Depot:

    x: float
    y: float
    z: float


@dataclass
class DemoUAV:

    x: float
    y: float
    z: float

    coverage_radius: float


@dataclass
class City:

    metadata: Metadata

    districts: List[District]

    roads: List[Road]

    buildings: List[Building]

    pois: List[PointOfInterest]

    nodes: List[IoTNode]

    depot: Depot

    demo_uav: DemoUAV


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def clip(value, lo, hi):

    return max(lo, min(value, hi))


def sample_inside_circle(cx, cy, radius):

    r = radius * np.sqrt(RNG.random())

    theta = RNG.uniform(0, 2 * np.pi)

    x = cx + r * np.cos(theta)

    y = cy + r * np.sin(theta)

    x = clip(x, 0, AREA_WIDTH)

    y = clip(y, 0, AREA_HEIGHT)

    return x, y


def sample_inside_building(building):

    x = RNG.uniform(

        building.x - building.width / 2,

        building.x + building.width / 2,

    )

    y = RNG.uniform(

        building.y - building.depth / 2,

        building.y + building.depth / 2,

    )

    return x, y


# =============================================================================
# METADATA
# =============================================================================

def generate_metadata():

    return Metadata(

        city_name=CITY_NAME,

        width=AREA_WIDTH,

        height=AREA_HEIGHT,

        seed=SEED,

    )


# =============================================================================
# DISTRICTS
# =============================================================================

def generate_districts():

    districts = []

    idx = 0

    for name, cfg in DISTRICT_CONFIG.items():

        districts.append(

            District(

                id=idx,

                name=name,

                center_x=cfg["center"][0],

                center_y=cfg["center"][1],

                radius=cfg["radius"],

                building_height_min=cfg["building_height"][0],

                building_height_max=cfg["building_height"][1],

                building_density=cfg["building_density"],

            )

        )

        idx += 1

    return districts


# =============================================================================
# ROADS
# =============================================================================

def generate_roads():

    roads = []

    rid = 0

    # ----------------------------
    # Main Horizontal Roads
    # ----------------------------

    for y in [250, 500, 750]:

        roads.append(

            Road(

                id=rid,

                x1=0,
                y1=y,

                x2=AREA_WIDTH,
                y2=y,

                road_type="main"

            )

        )

        rid += 1

    # ----------------------------
    # Main Vertical Roads
    # ----------------------------

    for x in [250, 500, 750]:

        roads.append(

            Road(

                id=rid,

                x1=x,
                y1=0,

                x2=x,
                y2=AREA_HEIGHT,

                road_type="main"

            )

        )

        rid += 1

    # ----------------------------
    # Minor Roads
    # ----------------------------

    for y in [125, 375, 625, 875]:

        roads.append(

            Road(

                id=rid,

                x1=0,
                y1=y,

                x2=AREA_WIDTH,
                y2=y,

                road_type="secondary"

            )

        )

        rid += 1

    for x in [125, 375, 625, 875]:

        roads.append(

            Road(

                id=rid,

                x1=x,
                y1=0,

                x2=x,
                y2=AREA_HEIGHT,

                road_type="secondary"

            )

        )

        rid += 1

    return roads


# =============================================================================
# POINTS OF INTEREST
# =============================================================================

def generate_pois():

    pois = [

        PointOfInterest(

            id=0,

            name="City Hospital",

            district="Hospital",

            x=500,

            y=180,

            category="hospital"

        ),

        PointOfInterest(

            id=1,

            name="Power Station",

            district="PowerStation",

            x=170,

            y=820,

            category="power"

        ),

        PointOfInterest(

            id=2,

            name="Industrial Control",

            district="Industrial",

            x=780,

            y=820,

            category="industrial"

        ),

        PointOfInterest(

            id=3,

            name="Business Center",

            district="Downtown",

            x=500,

            y=500,

            category="commercial"

        ),

    ]

    return pois


# =============================================================================
# BUILDING HELPERS
# =============================================================================

def nearest_poi(x, y, pois):

    best = None

    best_dist = 1e18

    for poi in pois:

        d = np.hypot(

            x - poi.x,

            y - poi.y

        )

        if d < best_dist:

            best_dist = d

            best = poi

    return best


# =============================================================================
# BUILDING GENERATOR
# =============================================================================

def generate_buildings(

    districts,

    pois,

    buildings_per_district=18

):

    buildings = []

    bid = 0

    for district in districts:

        for _ in range(buildings_per_district):

            x, y = sample_inside_circle(

                district.center_x,

                district.center_y,

                district.radius * 0.85

            )

            width = RNG.uniform(

                18,

                40

            )

            depth = RNG.uniform(

                18,

                40

            )

            height = RNG.uniform(

                district.building_height_min,

                district.building_height_max

            )

            poi = nearest_poi(

                x,

                y,

                pois

            )

            buildings.append(

                Building(

                    id=bid,

                    district=district.name,

                    poi=poi.name,

                    x=x,

                    y=y,

                    width=width,

                    depth=depth,

                    height=height

                )

            )

            bid += 1

    return buildings


# =============================================================================
# BUILDING LOOKUP
# =============================================================================

def buildings_of_district(

    buildings,

    district

):

    return [

        b

        for b in buildings

        if b.district == district

    ]


def buildings_of_poi(

    buildings,

    poi_name

):

    return [

        b

        for b in buildings

        if b.poi == poi_name

    ]


# =============================================================================
# DISTRICT LOOKUP
# =============================================================================

def district_by_name(

    districts,

    name

):

    for district in districts:

        if district.name == name:

            return district

    raise ValueError(f"Unknown district: {name}")


# =============================================================================
# POI LOOKUP
# =============================================================================

def poi_by_name(

    pois,

    name

):

    for poi in pois:

        if poi.name == name:

            return poi

    raise ValueError(f"Unknown POI: {name}")


# =============================================================================
# RANDOM BUILDING
# =============================================================================

def random_building(

    buildings,

    district=None,

    poi=None

):

    candidates = buildings

    if district is not None:

        candidates = [

            b

            for b in candidates

            if b.district == district

        ]

    if poi is not None:

        candidates = [

            b

            for b in candidates

            if b.poi == poi

        ]

    return RNG.choice(candidates)


# =============================================================================
# RANDOM POSITION INSIDE BUILDING
# =============================================================================

def random_position_on_building(

    building

):

    x = RNG.uniform(

        building.x - building.width / 2,

        building.x + building.width / 2,

    )

    y = RNG.uniform(

        building.y - building.depth / 2,

        building.y + building.depth / 2,

    )

    z = np.clip(

        building.height +

        RNG.normal(0, 1.0),

        0,

        40,

    )

    return x, y, z


# =============================================================================
# SUMMARY
# =============================================================================

def city_summary(

    city

):

    print()

    print("=" * 60)

    print(city.metadata.city_name)

    print("=" * 60)

    print(f"Districts : {len(city.districts)}")

    print(f"Roads     : {len(city.roads)}")

    print(f"Buildings : {len(city.buildings)}")

    print(f"POIs      : {len(city.pois)}")

    if hasattr(city, "nodes"):

        print(f"Nodes     : {len(city.nodes)}")

    print("=" * 60)

    print()


# =============================================================================
# PART 1 COMPLETE
# =============================================================================


# =============================================================================
# NODE GENERATION
# =============================================================================

def generate_nodes(buildings):

    nodes = []

    nid = 0

    # -------------------------------------------------------------------------
    # HIGH PRIORITY
    # -------------------------------------------------------------------------

    high_buildings = (

        buildings_of_poi(buildings, "City Hospital") +

        buildings_of_poi(buildings, "Power Station")

    )

    for _ in range(HIGH_NODES):

        b = RNG.choice(high_buildings)

        x, y, z = random_position_on_building(b)

        nodes.append(

            IoTNode(

                id=nid,

                district=b.district,

                building_id=b.id,

                x=x,

                y=y,

                z=z,

                priority="high",

                demand=RNG.uniform(1.5, 2.0),

                required_rate=38e6

            )

        )

        nid += 1

    # -------------------------------------------------------------------------
    # MEDIUM PRIORITY
    # -------------------------------------------------------------------------

    medium_buildings = (

        buildings_of_poi(buildings, "Industrial Control") +

        buildings_of_poi(buildings, "Business Center")

    )

    for _ in range(MEDIUM_NODES):

        b = RNG.choice(medium_buildings)

        x, y, z = random_position_on_building(b)

        nodes.append(

            IoTNode(

                id=nid,

                district=b.district,

                building_id=b.id,

                x=x,

                y=y,

                z=z,

                priority="medium",

                demand=RNG.uniform(0.8, 1.5),

                required_rate=25e6

            )

        )

        nid += 1

    # -------------------------------------------------------------------------
    # LOW PRIORITY
    # -------------------------------------------------------------------------

    low_buildings = (

        buildings_of_district(buildings, "Residential_A") +

        buildings_of_district(buildings, "Residential_B")

    )

    for _ in range(LOW_NODES):

        b = RNG.choice(low_buildings)

        x, y, z = random_position_on_building(b)

        nodes.append(

            IoTNode(

                id=nid,

                district=b.district,

                building_id=b.id,

                x=x,

                y=y,

                z=z,

                priority="low",

                demand=RNG.uniform(0.2, 0.8),

                required_rate=8e6

            )

        )

        nid += 1

    RNG.shuffle(nodes)

    return nodes


# =============================================================================
# NODE HELPERS
# =============================================================================

def nodes_of_priority(

    nodes,

    priority

):

    return [

        node

        for node in nodes

        if node.priority == priority

    ]


def nodes_of_district(

    nodes,

    district

):

    return [

        node

        for node in nodes

        if node.district == district

    ]


def count_nodes(nodes):

    print()

    print("Node Distribution")

    print("------------------------------")

    print(

        "High   :", len(nodes_of_priority(nodes, "high"))

    )

    print(

        "Medium :", len(nodes_of_priority(nodes, "medium"))

    )

    print(

        "Low    :", len(nodes_of_priority(nodes, "low"))

    )

    print()

# =============================================================================
# DEPOT
# =============================================================================

def generate_depot():

    return Depot(

        x=50.0,

        y=50.0,

        z=0.0,

    )


# =============================================================================
# DEMO UAV
# =============================================================================

def generate_demo_uav():

    return DemoUAV(

        x=500.0,

        y=500.0,

        z=100.0,

        coverage_radius=170.0,

    )


# =============================================================================
# SAVE / LOAD
# =============================================================================

def save_city(

    city,

    filename=CITY_FILE,

):

    filename = Path(filename)

    filename.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    with open(filename, "wb") as f:

        pickle.dump(

            city,

            f,

            protocol=pickle.HIGHEST_PROTOCOL,

        )


def load_city(

    filename=CITY_FILE,

):

    filename = Path(filename)

    with open(filename, "rb") as f:

        city = pickle.load(f)

    return city


# =============================================================================
# CITY EXISTS
# =============================================================================

def city_exists(

    filename=CITY_FILE,

):

    return Path(filename).exists()


# =============================================================================
# CREATE OR LOAD
# =============================================================================

def create_or_load_city(

    filename=CITY_FILE,

):

    if city_exists(filename):

        return load_city(filename)

    city = generate_city()

    save_city(city, filename)

    return city


# =============================================================================
# SIMPLE STATISTICS
# =============================================================================

def node_statistics(

    nodes,

):

    high = len(nodes_of_priority(nodes, "high"))

    medium = len(nodes_of_priority(nodes, "medium"))

    low = len(nodes_of_priority(nodes, "low"))

    print()

    print("=" * 50)

    print("IoT Node Statistics")

    print("=" * 50)

    print(f"Total Nodes : {len(nodes)}")

    print(f"High        : {high}")

    print(f"Medium      : {medium}")

    print(f"Low         : {low}")

    print("=" * 50)

    print()


# =============================================================================
# BUILDING STATISTICS
# =============================================================================

def building_statistics(

    buildings,

):

    print()

    print("=" * 50)

    print("Building Statistics")

    print("=" * 50)

    print(f"Total Buildings : {len(buildings)}")

    print()

    districts = sorted(

        {

            b.district

            for b in buildings

        }

    )

    for district in districts:

        count = len(

            buildings_of_district(

                buildings,

                district,

            )

        )

        print(f"{district:<18} : {count}")

    print("=" * 50)

    print()


# =============================================================================
# POI STATISTICS
# =============================================================================

def poi_statistics(

    pois,

):

    print()

    print("=" * 50)

    print("Points of Interest")

    print("=" * 50)

    for poi in pois:

        print(

            f"{poi.name:<22}"

            f" ({poi.category})"

        )

    print("=" * 50)

    print()
# =============================================================================
# CITY GENERATOR
# =============================================================================

def generate_city():

    metadata = generate_metadata()

    districts = generate_districts()

    roads = generate_roads()

    pois = generate_pois()

    buildings = generate_buildings(

        districts,

        pois,

    )

    nodes = generate_nodes(

        buildings,

    )

    depot = generate_depot()

    demo_uav = generate_demo_uav()

    city = City(

        metadata=metadata,

        districts=districts,

        roads=roads,

        buildings=buildings,

        pois=pois,

        nodes=nodes,

        depot=depot,

        demo_uav=demo_uav,

    )

    return city


# =============================================================================
# VERIFY CITY
# =============================================================================

def verify_city(city):

    assert len(city.districts) > 0

    assert len(city.roads) > 0

    assert len(city.buildings) > 0

    assert len(city.pois) > 0

    assert len(city.nodes) == TOTAL_NODES

    high = len(nodes_of_priority(city.nodes, "high"))

    medium = len(nodes_of_priority(city.nodes, "medium"))

    low = len(nodes_of_priority(city.nodes, "low"))

    assert high == HIGH_NODES

    assert medium == MEDIUM_NODES

    assert low == LOW_NODES


# =============================================================================
# BUILD CITY IF NEEDED
# =============================================================================

def get_city():

    if city_exists():

        return load_city()

    city = generate_city()

    verify_city(city)

    save_city(city)

    return city


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    city = get_city()

    city_summary(city)

    building_statistics(city.buildings)

    poi_statistics(city.pois)

    node_statistics(city.nodes)

    print("Depot")

    print("----------------------------")

    print(

        f"({city.depot.x:.1f}, "

        f"{city.depot.y:.1f}, "

        f"{city.depot.z:.1f})"

    )

    print()

    print("Demo UAV")

    print("----------------------------")

    print(

        f"Position : "

        f"({city.demo_uav.x:.1f}, "

        f"{city.demo_uav.y:.1f}, "

        f"{city.demo_uav.z:.1f})"

    )

    print(

        f"Coverage Radius : "

        f"{city.demo_uav.coverage_radius:.1f} m"

    )

    print()

    print("=" * 60)

    print("Synthetic city generated successfully.")

    print(f"Saved to : {CITY_FILE}")

    print("=" * 60)
