ST.FlameGraphModel = function () {

    var nodes_idx_by_name_;
    var width_scaler_ = 1;
    var color_map_ = {};

    //  This creates the tree content but puts it in a obj referenced format
    var index_nodes_by_name_ = function (nodes_item, agg, index_to_get) {

        var nos = {};
        var perftree = nodes_item[ index_to_get ].perftree;

        for (var i in perftree) {
            nos[i] = perftree[i][agg];
        }

        return nos;
    };


    function createTree(obj, key, level, agg) {

        //var mag = 80 / (Math.pow(2, level));
        var first_ydata = nodes_idx_by_name_[key];

        var mag = parseInt(first_ydata * width_scaler_);
        mag = mag < 0 ? 0 : mag;

        const node = {name: key, level, magnitude: mag};
        const children = obj[key];

        if (children && children.length > 0) {
            node.children = children.map(childKey => createTree(obj, childKey, level + 1, agg));
            node.magnitude = mag;
        }

        node.color = getSpot2Color_(node.name, node.children);
        color_map_[node.name] = node.color;

        return node;
    }


    function getSpot2Color_(name, children) {

        // Hash only the node name to ensure consistent colors across runs
        // Previously included children names which caused color changes between runs
        var colorH = spot2ColorHash(name);

        // Generate random values for red, green, and blue components
        // Combine components into a CSS color string
        return colorH;
    };

    function spot2ColorHash(text, alpha) {
        const reverseString = text.split("").reverse().join("")
        const hash = jQuery.md5(reverseString)

        // Use different parts of the hash for better color distribution
        const r = parseInt(hash.slice(0, 2), 16)
        const g = parseInt(hash.slice(8, 10), 16)
        const b = parseInt(hash.slice(16, 18), 16)

        // Ensure colors are vibrant by boosting low values
        const minBrightness = 80;
        const maxBrightness = 220;
        const normalizedR = minBrightness + (r / 255) * (maxBrightness - minBrightness);
        const normalizedG = minBrightness + (g / 255) * (maxBrightness - minBrightness);
        const normalizedB = minBrightness + (b / 255) * (maxBrightness - minBrightness);

        return `rgb(${Math.round(normalizedR)}, ${Math.round(normalizedG)}, ${Math.round(normalizedB)}, 0.6)`
    }


    var get_ = function (ef, index_to_get, agg) {

        nodes_idx_by_name_ = index_nodes_by_name_(ef.nodes, agg, index_to_get);

        // Use per-run childrenMap if available, otherwise fall back to global
        var cm = ef.childrenMap;  // Default to global childrenMap

        if (ef.nodes && ef.nodes[index_to_get] && ef.nodes[index_to_get].childrenMap) {
            // Use the specific run's childrenMap
            cm = ef.nodes[index_to_get].childrenMap;
        } else if( !ef.childrenMap ) {
            console.log('************** Warning: I do not have a childrenMap. ********************');
        }

        const root = Object.keys(cm)[0];
        var topWidth = 620;

        if( nodes_idx_by_name_[root] ) {
            topWidth = nodes_idx_by_name_[root];
        }

        var flameContainerWidth = $('.flameContainer').width();

        var calc_width_scale = flameContainerWidth / topWidth;
        width_scaler_ = calc_width_scale;

        //  debug code.
        var tmp_first_ydata = nodes_idx_by_name_[root];

        const tree0 = createTree(cm, root, 0, agg);

        return {
            tree: tree0,
            max: nodes_idx_by_name_['main']
        }
    };


    var get_color_by_name_ = function( node_name ) {
        return color_map_[node_name] || "#eeeeee";
    };


    return {
        get_color_by_name: get_color_by_name_,
        get: get_
    }
}();
