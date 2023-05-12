"""
Module implementing the VadereRunner Class, which allows the execution of scenarios in Vadere through Python.

author: Simon Blöchinger
"""

import typing
import subprocess
import os


class VadereRunner:
    """A class allowing the execution of scenarios in Vadere through Python."""

    def __init__(self, vadere_dir: str, java_dir: typing.Optional[str] = "",
                 scenario_dir: typing.Optional[str] = None,
                 output_dir: typing.Optional[str] = None) -> None:
        """
        Initialize the Vadere Runner, which allows the user to run vadere scenarios from python.

        :param vadere_dir: The directory of the vadere executable.
        :param java_dir: The directory of the java executable, should be set if it is not in path.
        :param scenario_dir: The directory of the scenarios.
            If no directory is chosen, the full path will have to be selected for each scenario.
        :param output_dir: The directory where the outputs should be stored.
        """
        self.vadere_dir = vadere_dir
        self.java_dir = java_dir

        self.scenario_dir = scenario_dir
        self.output_dir = output_dir

        self.vadere_jar = os.path.join(self.vadere_dir, "vadere-console.jar")
        self.java_exe = os.path.join(self.java_dir, "java")

    def run_scenario(self, scenario_file: str, capture_output: bool = False) -> subprocess.CompletedProcess:
        """
        Run a scenario in vadere.

        :param scenario_file: The scenario file that will be executed in vadere.
            If no scenario_dir was selected for the class object, the whole path needs to be given.
            If a scenario_dir was selected in the class object,
            only the file name of the scenario file needs to be given.
        :param capture_output: If true, the Vadere console output will be captured and returned by this function.
            If false, the vadere console output will be printed.
        :return: Object containing the arguments used to start the process and the return value.
            If capture_output was true, it also contains the Vadere console output (stdout and stderr).
        """
        if self.scenario_dir is not None:
            scenario_file = os.path.join(self.scenario_dir, scenario_file)

        command = [self.java_exe, '-jar', self.vadere_jar, 'scenario-run', '--scenario-file',
                   scenario_file]
        if self.output_dir is not None:
            command += ["--output-dir", self.output_dir]

        completed_process = subprocess.run(command, capture_output=capture_output)
        return completed_process
