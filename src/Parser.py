import sys
from pathlib import Path
from typing import Optional
from configparser import ConfigParser, NoSectionError, NoOptionError
import xml.etree.ElementTree as ET
from numpy.typing import NDArray
from Solid import define_material_properties

class XMLHandler:
    """A class for handling XML file operations including parsing, searching, and modification.
    
    Attributes:
        filepath: Path to the XML file
        tree: Parsed XML tree structure
        root: Root element of the XML tree
    """

    def __init__(self, filepath: Path) -> None:
        """Initialize the XMLHandler with a file path.
        
        Args:
            filepath: Path to the XML file to handle
        """
        self.filepath = filepath
        self.tree: ET.ElementTree = ET.ElementTree()
        self.root: Optional[ET.Element] = None
        self.load_xml()

    def load_xml(self) -> None:
        """Load and parse the XML file."""
        try:
            self.tree = ET.parse(self.filepath)
            self.root = self.tree.getroot()
        except ET.ParseError as e:
            print(f"ERROR: Failed to parse XML file {self.filepath}: {str(e)}!")
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"ERROR: XML file {self.filepath} not found!")
            sys.exit(1)

    def print_xml_content(self) -> None:
        """Recursively print the entire XML structure with indentation."""
        def recursive_print(element: ET.Element, indent: str = "") -> None:
            """Helper function for recursive printing of XML elements."""
            print(f"{indent}Tag: {element.tag}, Attributes: {element.attrib}")
            if element.text and element.text.strip():
                print(f"{indent}Text: {element.text.strip()}")
            for child in element:
                recursive_print(child, indent + "  ")
        
        if self.root is not None:
            recursive_print(self.root)
        else:
            print("No XML content loaded")

    def save_xml(self, output_filepath: Path) -> None:
        """Save the XML tree to a file.
        
        Args:
            output_filepath: Path to save the XML file
        """
        try:
            self.tree.write(output_filepath, encoding="utf-8", xml_declaration=True)
            print(f"XML file successfully saved to {output_filepath}.")
        except IOError as e:
            print(f"ERROR: Failed to save XML file to {output_filepath}: {str(e)}!")
            sys.exit(1)

    def find_elements(self, node_tag: str, display: bool = False) -> Optional[list[ET.Element]]:
        """Find all elements with a specific tag in the XML tree.
        
        Args:
            node_tag: The tag name to search for
            display: Whether to print found elements
            
        Returns:
            list of matching elements or [] if none found
        """
        if self.root is None:
            return None
            
        elements = self.root.findall(node_tag)
        if elements:
            if display:
                for element in elements:
                    element_str = ET.tostring(element, encoding="utf-8").decode("utf-8")
                    print(element_str)
            return elements
        else:
            if display:
                print(f"No elements found with tag '{node_tag}'")
            return []

    def find_elem_with_attrib(self, node: ET.Element, attribute: str) -> dict[str, str]:
        """Find a child element with a specific attribute within a node.
        
        Args:
            node: Parent element to search within
            attribute: Attribute name to search for
            
        Returns:
            dictionary of the element's attributes
        """
        for elem in node:
            if attribute == elem.tag:
                return elem.attrib
        print(f"ERROR: Attribute '{attribute}' not found in node {node.tag}!")
        sys.exit(1)

    def find_all_attrib(self, attrib: str) -> list[str]:
        """Find all unique values for a specific attribute in the XML tree.
        
        Args:
            attrib: Attribute name to search for
            
        Returns:
            list of unique attribute values
        """
        if self.root is None:
            return []
            
        all_attribs = []
        for elem in self.root.iter():
            if attrib in elem.attrib:
                all_attribs.append(elem.attrib[attrib])
        return list(set(all_attribs))
        
    def add_element_with_attribute(
        self,
        attribute: str,
        value: str,
        add_attribute: str,
        add_value: str
    ) -> None:
        """Add an attribute to an element identified by an existing attribute.
        
        Args:
            attribute: Attribute name to identify the target element
            value: Attribute value to identify the target element
            add_attribute: New attribute name to add
            add_value: New attribute value to set
            
        Returns:
            None
        """
        target_node = None
        if self.root is None:
            print("No XML content loaded")
            return
            
        for node in self.root.iter():
            if attribute in node.attrib and node.attrib[attribute] == value:
                target_node = node
                break
                
        if target_node is not None:
            target_node.set(add_attribute, add_value)
            print(f"Attribute '{add_attribute}' successfully set to '{add_value}'")
            self.save_xml(self.filepath)
        else:
            print(f"No node found with {attribute}='{value}'")

    def add_element_with_custom_name(
        self,
        tag_name: str,
        old_name: str,
        new_name: str
    ) -> None:
        """Create a new element by copying an existing one with a new name.
        
        Args:
            tag_name: Tag name of elements to search for
            old_name: Current 'name' attribute value to match
            new_name: New 'name' attribute value to set
            
        Returns:
            None
        """
        if self.root is None:
            print("No XML content loaded")
            return
            
        for element in self.root.findall(tag_name):
            if element.get('name') == old_name:
                new_element = ET.Element(tag_name)
                for attr, val in element.items():
                    new_element.set(attr, new_name if attr == 'name' else val)
                self.root.append(new_element)
        self.save_xml(self.filepath)

def read_files(
    config_file: Path
) -> tuple[dict[str, XMLHandler], ConfigParser, ConfigParser]:
    """Read and validate XML and configuration files.
    
    Args:
        config_file: Path to the simulation configuration file
        
    Returns:
        - List of XMLHandler instances
        - Material library config parser
        - Simulation config parser
    """

    # Read simulation configuration
    try:
        sim_config = ConfigParser()
        if not sim_config.read(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} not found!")
    except Exception as e:
        print(f"ERROR: {str(e)}!")
        sys.exit(1)

    # Read and parse XML files
    try:
        xml_files = sim_config.get('Structure', 'path')
        xml_files = xml_files.split(',')
        xml_files = [Path(xml_file.strip()) for xml_file in xml_files if xml_file.strip()]
    except Exception as e:
        print(f"ERROR: {str(e)}!")
        sys.exit(1)

    system_xml_dict: dict[str, XMLHandler] = {}
    try:
        for xml_file in xml_files:
            system_xml_dict[xml_file.stem] = XMLHandler(xml_file)
    except Exception as e:
        print(f"ERROR loading XML file {xml_file}: {str(e)}")
        sys.exit(1)
    
    # Read and parse material library
    try:
        matlib_file = sim_config.get('Matlib', 'path')
    except Exception as e:
        print(f"ERROR: {str(e)}!")
        sys.exit(1)
    
    try:
        mat_lib = ConfigParser()
        if not mat_lib.read(matlib_file):
            raise FileNotFoundError(f"Material library {matlib_file} not found!")
    except Exception as e:
        print(f"ERROR: {str(e)}!")
        sys.exit(1)

    # Validate material properties
    list_of_labels = []
    for system_xml in system_xml_dict.values():
        list_of_labels.extend(system_xml.find_all_attrib('FillingMaterial'))
        if system_xml.find_elements("NoPackage"):
            list_of_labels.append('NoPackage')

    for label in list_of_labels:
        try:
            material_type = mat_lib.get(label, 'type')
        except NoSectionError:
            print(f"ERROR: Material '{label}' not defined!")
            sys.exit(1)
        except NoOptionError:
            print(f"ERROR: Material type not defined for label '{label}'!")
            sys.exit(1)

        try:
            required_properties = sim_config[material_type]['properties'].split(',')
            required_properties = [prop.strip() for prop in required_properties if prop.strip()]
        except:
            print(f"ERROR: Material type '{material_type}' not defined in simulation config")
            sys.exit(1)

        missing_props = []
        for prop in required_properties:
            if not mat_lib.has_option(label, prop):
                missing_props.append(prop)

        if missing_props:
            print(f"ERROR: Material '{label}' of type '{material_type}' is missing properties: {', '.join(missing_props)}")
            sys.exit(1)

    return system_xml_dict, sim_config, mat_lib


class DataCollection:
    def __init__(self, config: object, matlib: object) -> None:
        """
        Initialize the DataCollection with configuration, material library, and power library.
        
        Args:
            config: Configuration file object.
            matlib: Material library configuration.
        """
        self.config = config
        self.matlib = matlib

        try:
            # Simulation configuration
            self.sim_cfg = config._sections["Simulation"]
            self.mode = int(self.sim_cfg["mode"])
            self.interp_num = int(self.sim_cfg["interp_num"])
            model_dir = self.sim_cfg["model_dir"]
            if model_dir:
                self.model_dir = Path(model_dir)
            else:
                self.model_dir = Path('./model')
            result_dir = self.sim_cfg["result_dir"]
            if result_dir:
                self.result_dir = Path(result_dir)
            else:
                self.result_dir = Path('./raw')

            # Offline Mesh configuration
            self.mesh_dict = config._sections["MeshConfig"]
            self.mesh_size = float(self.mesh_dict["mesh_size"])
            self.unit = int(self.mesh_dict["unit"])
            self.precision = int(self.mesh_dict["precision"])
            self.eps = 10**(self.unit - self.precision - 1)

            # Online Solver configuration
            self.solver_dict = config._sections["SolverConfig"]
            self.solver_type = int(self.solver_dict["solver_type"])
            if self.solver_type == 0:
                self.rtol = 0.0
                self.atol = 0.0
                self.max_it = 10000
                try:
                    self.rtol = float(self.solver_dict["rtol"])
                except:
                    try:
                        self.atol = float(self.solver_dict["atol"])
                    except:
                        raise KeyError("rtol or atol")
                try:
                    self.max_it = int(self.solver_dict["max_it"])
                except:
                    raise KeyError("max_it")
            self.is_display = False
            if self.solver_dict["is_display"].lower() == "true":
                self.is_display = True

            # Temperature configuration
            self.temp_cfg = config._sections["Temperature"]
            self.ref_temp = float(self.temp_cfg["reference"])
            self.amb_temp = float(self.temp_cfg["ambient"])

            # Structure configuration
            self.struct_cfg = config._sections["Structure"]
            self.default_name: str = self.struct_cfg["default"]
            xml_paths = self.struct_cfg["path"]
            xml_paths = xml_paths.split(',')
            xml_paths = [Path(xml_path.strip()) for xml_path in xml_paths if xml_path.strip()]
            self.structure_names = [xml_path.stem for xml_path in xml_paths]

            # Floorplan configuration
            self.flp_cfg = config._sections["Floorplan"]
            self.flp_path = Path(self.flp_cfg["path"])

            # Boundary configruation
            self.boundary_cfg = config._sections["Boundary"]
            origin = self.boundary_cfg["origin"]
            try:
                x_str, y_str, z_str = origin.split(",")
                self.x_origin: float = float(x_str.strip())
                self.y_origin: float = float(y_str.strip())
                self.z_origin: float = float(z_str.strip())
            except:
                pass
            temp_path = self.boundary_cfg["temp_path"]
            if temp_path:
                self.temp_path = Path(temp_path)
            else:
                self.temp_path = None
            disp_path = self.boundary_cfg["disp_path"]
            if disp_path:
                self.disp_path = Path(disp_path)
            else:
                self.disp_path = None
            
            # Postprocessing configuration
            self.post_cfg = config._sections["Postprocess"]
            value_names = self.post_cfg["value"]
            value_names = value_names.split(',')
            self.value_names: list[str] = [value_name.strip() for value_name in value_names if value_name.strip()]
            self.reso = int(self.post_cfg["reso"])
            self.z_coord = float(self.post_cfg["z_coord"])

        except Exception as e:
            print(f"ERROR: Missing configuration parameter: {str(e)}!")
            sys.exit(1)

        # Material properties
        self.mat_dict :dict[str, dict[str, NDArray]] = {}
        for section in matlib.sections():
            self.mat_dict[section] = define_material_properties(section, matlib)
        