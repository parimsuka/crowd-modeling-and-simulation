import typing
import subprocess
import os


class VadereRunner:
    def __init__(self, vadere_dir: str, java_dir: typing.Optional[str] = "",
                 scenario_dir: typing.Optional[str] = None,
                 output_dir: typing.Optional[str] = None) -> None:
        self.vadere_dir = vadere_dir

        # Directionary for the java executable, should be set if it is not in path
        self.java_dir = java_dir

        self.scenario_dir = scenario_dir
        self.output_dir = output_dir

        self.vadere_jar = os.path.join(self.vadere_dir, "vadere-console.jar")
        self.java_exe = os.path.join(self.java_dir, "java")

    def run_scenario(self, scenario_file: str, capture_output=True):
        if self.scenario_dir is not None:
            scenario_file = os.path.join(self.scenario_dir, scenario_file)

        # command = f"{self.java_exe} -jar {self.vadere_jar} scenario-run " \
        #           f"--scenario-file \"{scenario_file}\""
        #
        # if self.output_dir is not None:
        #     command += f" --output-dir \"{self.output_dir}\""

        command = [self.java_exe, '-jar', self.vadere_jar, 'scenario-run', '--scenario-file',
                   scenario_file]
        if self.output_dir is not None:
            command += ["--output-dir", self.output_dir]

        print(command)

        completed_process = subprocess.run(command, capture_output=capture_output)
        return completed_process
