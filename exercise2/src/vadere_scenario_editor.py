"""
Module implementing the VadereScenarioEditor Class, which allows editing Vadere scenarios through Python.
Implements two functions update_dict_smart and get_values_for_key which are used in the editing process.

author: Simon Blöchinger
"""

import json
import typing


def update_dict_smart(dictionary: dict, update_tuple: tuple) -> None:
    """
    Allows the updating or creation of any key inside a nested dictionary by reading a possibly nested update_tuple.

    :param dictionary: The dictionary object that will be edited (in place).
    :param update_tuple: A (possibly recursive) tuple of the format `(key, value)` that will be used to update
        `key` inside `dictionary` with `value`. `value` can be another tuple.
    """
    key, value = update_tuple
    if isinstance(value, tuple):
        # If `value` is another tuple: Enter `key` and re-run the method.
        return update_dict_smart(dictionary[key], value)
    else:
        # Otherwise, just edit `key` with `value`.
        dictionary[key] = value


def get_values_for_key(search_object: any, search_key: str) -> list:
    """
    A function that searches a nested dictionary and returns all values found for a specific key at any level.

    :param search_object: The dictionary (or list) that will be searched.
    :param search_key: The key that will be searched for.
    :return: A list of all values found inside the search_object for the search_key.
    """
    r = []

    # Test: search_object is List
    if isinstance(search_object, list):
        for list_item in search_object:
            # Try to search deeper for every value, in case it is a nested dictionary (or list)
            r += get_values_for_key(list_item, search_key)

    # Test: search_object is Dictionary
    if isinstance(search_object, dict):
        for key, value in search_object.items():
            if key == search_key:
                # If the search_key is found: add the value to our return list.
                r.append(value)
            # Try to search deeper for every value, in case it is a nested dictionary (or list)
            r += get_values_for_key(value, search_key)

    return r


class VadereScenarioEditor:
    """
    A class implementing the Vadere Scenario Editor, which allows easy insertion of pedestrian
    into scenarios, but also offers flexible functions to edit other scenario attributes.
    """
    def __init__(self, scenario_file_path: str, allow_overwriting: bool = False) -> None:
        """
        Initializes the VadereScenarioEditor.

        :param scenario_file_path: The path to the scenario file that will be edited.
        :param allow_overwriting: If true: allows saving the new scenario file at in scenario_file_path.
        """
        self.allow_overwriting = allow_overwriting

        # Load the scenario
        self.scenario_file_path = None  # Will be overwritten by self.load_scenario
        self.scenario = None  # Will be overwritten by self.load_scenario
        self.load_scenario(scenario_file_path)

    def get_scenario(self) -> dict:
        """
        Returns the scenario that is currently edited by the VadereScenarioEditor as a dictionary.

        :return: Dictionary containing the currently edited scenario.
        """
        return self.scenario

    def get_scenario_string(self) -> str:
        """
        Returns the scenario that is currently edited by the VadereScenarioEditor as a json string.

        :return: String containing the currently edited scenario.
        """
        return json.dumps(self.scenario, indent=2)

    def save_scenario(self, save_file_path: typing.Optional[str] = None) -> None:
        """
        Saves the currently edited scenario to a file.

        :param save_file_path: The file that the currently edited scenario is written to.
            If no file is given and self.allow_overwriting is True, the path that the currently edited
            scenario was read from is used and the file overwritten.

        :raises:
            ValueError: No save_file_path was specified, but self.allow_overwriting is False.
        """
        if save_file_path is None:
            if self.allow_overwriting:
                save_file_path = self.scenario_file_path
            else:
                raise ValueError("No save_file_path was specified, but overwriting is not allowed.")

        with open(save_file_path, 'w', encoding='utf-8') as save_file:
            json.dump(self.scenario, save_file, ensure_ascii=False, indent=2, separators=(',', ' : '))

    def load_scenario(self, open_file_path: typing.Optional[str] = None) -> None:
        """
        Loads a scenario that will be edited by the VadereScenarioEditor.

        :param open_file_path: The file path of the scenario that will be loaded.
            If None: Reloads the current scenario from self.scenario_file_path.
        """
        if open_file_path is None:
            # reload the current scenario file
            open_file_path = self.scenario_file_path

        with open(open_file_path, 'r') as open_file:
            scenario = json.load(open_file)
        self.scenario = scenario
        self.scenario_file_path = open_file_path

    def edit_scenario(self, key_word_args: tuple) -> None:
        """
        Allows flexibly editing the scenario by specifying key_word_args in the form of `(key, value)`

        :param key_word_args: Tuple containing a key and a value. If value is another key-value-tuple,
            scenario['key'] will be edited with this key-value-tuple.
        """
        update_dict_smart(self.scenario, key_word_args)

    def insert_pedestrian(self, x: float, y: float, target_ids: list[int] = None,
                          identifier: typing.Optional[int] = None,
                          width: float = 1, height: float = 1,
                          key_word_args: typing.Optional[list[tuple[str, any]]] = None) -> None:
        """
        Inserts a pedestrian into the scenario.

        :param x: The x-coordinate of the pedestrian.
        :param y: The y-coordinate of the pedestrian.
        :param target_ids: A list of target_ids that the pedestrian will consider as targets.
            If None: an empty list will be added as target_ids for the pedestrian.
        :param identifier: The id of the pedestrian. If None: The smallest free id will be selected.
        :param width: The width of the pedestrian.
        :param height: The height of the pedestrian.
        :param key_word_args: A list of tuples that is used to further edit the inserted pedestrian.
            The tuples are `(key, value)` pairs, where `key` will be updated in the pedestrian with `value`.
            If `value` is another key-value-tuple, pedestrian['key'] will be updated with this second tuple.
        """
        # If no target_ids list is specified: add an empty list.
        if target_ids is None:
            target_ids = []

        # If no id is specified: search for the first free id
        if identifier is None:
            identifier = self._get_valid_id_for_pedestrian()

        # The default pedestrian object with common attributes edited
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

        # Edit further (less common) attributes if there are any given
        if key_word_args is not None:
            for kwa in key_word_args:
                update_dict_smart(pedestrian_object, kwa)

        self.scenario['scenario']['topography']['dynamicElements'].append(pedestrian_object)

    def _get_valid_id_for_pedestrian(self) -> int:
        """
        Searches for the first free ID inside scenario.topography, which can be assigned to a pedestrian.

        :return: The first free ID inside scenario.topography.
        """
        # Get all ids that are already in use inside scenario.topography
        used_ids = get_values_for_key(self.scenario['scenario']['topography'], "id")
        identifier = 1  # Vadere starts identifiers from 1
        # Search for the first free id
        while identifier in used_ids:
            identifier += 1
        return identifier
