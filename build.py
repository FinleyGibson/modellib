#   -*- coding: utf-8 -*-
from pybuilder.core import use_plugin, init, task, depends

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")
use_plugin("python.install_dependencies")


name = "modellib"
default_task = "publish"


@init
def set_properties(project):
    project.set_property("coverage_break_build", True)
    project.build_depends_on("parameterized")
    project.depends_on("numpy")



@task
def example_task(logger):
    print("Example task running")
