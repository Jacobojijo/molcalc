import datetime
import hashlib
import logging
import re

import models
import numpy as np
import pipelines
from pyramid import httpexceptions
from pyramid.view import notfound_view_config, view_config
from rdkit import Chem
from rdkit.Chem import AllChem

from molcalc import constants
from molcalc_lib import gamess_results
from ppqm import chembridge

_logger = logging.getLogger("molcalc:views")

# Error Views
@notfound_view_config(renderer="templates/page_404.html")
def not_found(request):
    """
    404 Not Found view
    """
    request.response.status = 404
    return {}

# Calculation Views
@view_config(route_name="editor", renderer="templates/page_editor.html")
def editor(request):
    """
    Standard view for MolCalc. Static HTML.
    """
    return {}

@view_config(route_name="calculation", renderer="templates/page_calculation.html")
def view_calculation(request):
    """
    View for looking up calculations.
    """
    hashkey = request.matchdict["one"]

    # Look up the key
    calculation = request.dbsession.query(models.GamessCalculation).filter_by(hashkey=hashkey).first()

    if not calculation or hashkey == "404":
        raise httpexceptions.exception_response(404)

    data = gamess_results.view_gamess_calculation(calculation)
    return data

@view_config(route_name="calculations", renderer="templates/page_calculation.html")
def view_calculations(request):
    """
    Statistic about current calculations? Iono, maybe not.
    """
    raise httpexceptions.exception_response(404)
    return {}

# Static page views
@view_config(route_name="about", renderer="templates/page_about.html")
def about(request):
    """
    Static about page
    """
    return {}

@view_config(route_name="help", renderer="templates/page_help.html")
def page_help(request):
    """
    Static help page
    """
    return {}

@view_config(route_name="sdf_to_smiles", renderer="json")
def ajax_sdf_to_smiles(request):
    """
    SDF to SMILES conversion
    """
    if not request.POST:
        return {"error": "Error 55 - Missing key", "message": "Error. Missing information."}

    try:
        sdf = request.POST["sdf"].encode("utf-8")
    except Exception:
        return {"error": "Error 60 - get error", "message": "Error. Missing information."}

    smiles, status = chembridge.sdf_to_smiles(sdf)
    if smiles is None:
        status = status.split("]")[-1]
        status = re.sub(r"\# [0-9]+", "", status)
        return {"error": "Error 69 - rdkit error", "message": status}

    return {"smiles": smiles}

# Ajax views
@view_config(route_name="smiles_to_sdf", renderer="json")
def ajax_smiles_to_sdf(request):
    """
    Convert SMILES to SDF format
    """
    if not request.POST:
        return {"error": "Error 53 - Missing key", "message": "Error. Missing information."}

    try:
        smiles = request.POST["smiles"].encode("utf-8")
    except Exception as e:
        return {"error": "Error 58 - get error", "message": "Error. Missing information.", "exception": f"{e}"}

    sdfstr = chembridge.smiles_to_sdfstr(smiles)
    return {"sdf": sdfstr}

@view_config(route_name="submitquantum", renderer="json")
def ajax_submitquantum(request):
    """
    Setup quantum calculation
    """
    settings = request.registry.settings
    user_ip = request.remote_addr

    # Check if user is a known misuser
    if constants.COLUMN_BLOCK_IP in settings and user_ip in settings[constants.COLUMN_BLOCK_IP]:
        return {"error": "Error 194 - blocked ip", "message": "IP address has been blocked for missue"}

    if not request.POST:
        return {"error": "Error 128 - empty post", "message": "Error. Empty post."}

    if not request.POST["sdf"]:
        return {"error": "Error 132 - sdf key error", "message": "Error. Missing information."}

    sdfstr = request.POST["sdf"].encode("utf-8")
    add_hydrogens = request.POST.get("add_hydrogens", "1") == "1"

    molobj, status = chembridge.sdfstr_to_molobj(sdfstr, return_status=True)
    if molobj is None:
        status = status.split("]")[-1]
        status = re.sub(r"\# [0-9]+", "", status)
        return {"error": "Error 141 - rdkit error", "message": status}

    try:
        molobj.GetConformer()
    except ValueError:
        return {"error": "Error 141 - rdkit error", "message": "Error. Server was unable to generate conformations for this molecule"}

    atoms = chembridge.molobj_to_atoms(molobj)
    if 1 not in atoms and add_hydrogens:
        molobj = Chem.AddHs(molobj)
        AllChem.EmbedMultipleConfs(molobj, numConfs=1)
        chembridge.molobj_optimize(molobj)

    atoms = chembridge.molobj_to_atoms(molobj)
    max_atoms = 10
    heavy_atoms = np.where(atoms != 1)[0]
    if len(heavy_atoms) > max_atoms:
        return {"error": "Error 194 - max atoms error", "message": "Stop Casper. Max ten heavy atoms."}

    sdfstr = sdfstr.decode("utf8")
    for _ in range(3):
        sdfstr = sdfstr[sdfstr.index("\n") + 1:]
    sdfstr = "\n" * 3 + sdfstr

    hashkey = hashlib.md5(sdfstr.encode()).hexdigest()

    calculation = request.dbsession.query(models.GamessCalculation).filter_by(hashkey=hashkey).first()

    if calculation is not None:
        msg = {"hashkey": hashkey}
        calculation.created = datetime.datetime.now()
        _logger.info(f"{hashkey} exists")
        return msg

    _logger.info(f"{hashkey} create")
    molecule_info = {"sdfstr": sdfstr, "molobj": molobj, "hashkey": hashkey}

    try:
        msg, new_calculation = pipelines.calculation_pipeline(molecule_info, settings)
    except Exception:
        sdfstr = chembridge.molobj_to_sdfstr(molobj)
        _logger.error(f"{hashkey} PipelineError", exc_info=True)
        _logger.error(sdfstr)
        return {"error": "293", "message": "Internal server server. Uncaught exception"}

    if new_calculation is not None:
        request.dbsession.add(new_calculation)

    return msg