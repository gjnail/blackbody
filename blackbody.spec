# PyInstaller build: `pyinstaller blackbody.spec` -> dist/Blackbody/ with Blackbody.exe (app) and
# blackbody-cli.exe (command line for batch and render-farm use).
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

datas = [
    ('blackbody/engine/wgsl/*.wgsl', 'blackbody/engine/wgsl'),
    ('blackbody/assets/presets/*.png', 'blackbody/assets/presets'),
    ('blackbody/assets/meshes/*.obj', 'blackbody/assets/meshes'),
    ('blackbody/assets/blackbody.png', 'blackbody/assets'),
]
datas += collect_data_files('wgpu')
binaries = (collect_dynamic_libs('wgpu') + collect_dynamic_libs('av') + collect_dynamic_libs('OpenEXR')
            + collect_dynamic_libs('mujoco'))
hidden = collect_submodules('blackbody') + collect_submodules('wgpu.backends') + ['OpenEXR', 'mujoco']
excludes = ['tkinter', 'matplotlib', 'scipy', 'pandas', 'IPython', 'pytest']

a = Analysis(['tools/launch_gui.py'], pathex=['.'], binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
app = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Blackbody', console=False,
          icon='blackbody/assets/blackbody.ico')
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name='blackbody-cli', console=True,
          icon='blackbody/assets/blackbody.ico')
coll = COLLECT(app, cli, a.binaries, a.datas, name='Blackbody')
