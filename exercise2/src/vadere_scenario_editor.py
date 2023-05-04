import json
import typing


def update_dict_smart(dictionary: dict, update_tuple: tuple):
    key, value = update_tuple
    if type(value) is tuple:
        update_dict_smart(dictionary[key], value)
    else:
        dictionary[key] = value
        # dictionary.update((key, value))


class VadereScenarioEditor:
    def __init__(self, scenario_file_path: str, allow_overwriting: bool = False) -> None:
        self.scenario_file_path = scenario_file_path
        self.allow_overwriting = allow_overwriting

        # Load the scenario
        self.scenario = None
        self.load_scenario(scenario_file_path)

    def get_scenario(self) -> dict:
        return self.scenario

    def get_scenario_string(self) -> str:
        return json.dumps(self.scenario, indent=2)

    def save_scenario(self, save_file_path: typing.Optional[str] = None) -> None:

        if save_file_path is None:
            if self.allow_overwriting:
                save_file_path = self.scenario_file_path
            else:
                raise ValueError("No save_file_path was specified, but overwriting is not allowed.")

        with open(save_file_path, 'w', encoding='utf-8') as save_file:
            json.dump(self.scenario, save_file, ensure_ascii=False, indent=2,
                      separators=(',', ' : '))

    def load_scenario(self,
                      open_file_path: typing.Optional[str] = None) -> None:
        if open_file_path is None:
            # reload the current scenario file
            open_file_path = self.scenario_file_path

        with open(open_file_path, 'r') as open_file:
            scenario = json.load(open_file)
        self.scenario = scenario

    def edit_scenario(self, key_word_args):
        update_dict_smart(self.scenario, key_word_args)

    def insert_pedestrian(self, x: float, y: float, target_ids: list[int] = None,
                          identifier: int = -1, width: float = 1,
                          height: float = 1,
                          key_word_args: typing.Optional[list[tuple[str, any]]] = None):
        if target_ids is None:
            target_ids = []

        pedestrian_object = {
            "attributes": {
                "id": identifier,
                "shape": {
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": height,
                    "type": "RECTANGLE"
                },
                "visible": True,
                "radius": 0.2,
                "densityDependentSpeed": False,
                "speedDistributionMean": 1.34,
                "speedDistributionStandardDeviation": 0.26,
                "minimumSpeed": 0.5,
                "maximumSpeed": 2.2,
                "acceleration": 2.0,
                "footstepHistorySize": 4,
                "searchRadius": 1.0,
                "walkingDirectionSameIfAngleLessOrEqual": 45.0,
                "walkingDirectionCalculation": "BY_TARGET_CENTER"
            },
            "source": None,
            "targetIds": target_ids,
            "nextTargetListIndex": 0,
            "isCurrentTargetAnAgent": False,
            "position": {
                "x": x,
                "y": y
            },
            "velocity": {
                "x": 0.0,
                "y": 0.0
            },
            "freeFlowSpeed": 1.3330991286089942,
            "followers": [],
            "idAsTarget": -1,
            "isChild": False,
            "isLikelyInjured": False,
            "psychologyStatus": {
                "mostImportantStimulus": None,
                "threatMemory": {
                    "allThreats": [],
                    "latestThreatUnhandled": False
                },
                "selfCategory": "TARGET_ORIENTED",
                "groupMembership": "OUT_GROUP",
                "knowledgeBase": {
                    "knowledge": [],
                    "informationState": "NO_INFORMATION"
                },
                "perceivedStimuli": [],
                "nextPerceivedStimuli": []
            },
            "healthStatus": None,
            "infectionStatus": None,
            "groupIds": [],
            "groupSizes": [],
            "agentsInGroup": [],
            "trajectory": {
                "footSteps": []
            },
            "modelPedestrianMap": None,
            "type": "PEDESTRIAN"
        }

        if key_word_args is not None:
            for kwa in key_word_args:
                update_dict_smart(pedestrian_object, kwa)

        self.scenario['scenario']['topography']['dynamicElements'].append(pedestrian_object)
