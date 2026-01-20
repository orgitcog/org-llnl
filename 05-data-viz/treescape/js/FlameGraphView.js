var ST = ST || {};

ST.FlameGraphView = function () {

    var tree_;

    var renderBar_ = function (leaf) {
        var width = leaf.magnitude; //leaf.width * scaling_factor;
        var color = leaf.color; //getSpot2Color_(leaf.name, leaf.children);
        return '<div style="width: ' + width + 'px; background-color: ' + color + '" class="block">' +
            '<div class="text">' + leaf.name + '</div>' +
            '</div>';
    };


    var renderLeaf_ = function (leaf) {

        var children_ht = "";

        if (leaf.children) {

            for (var x = 0; x < leaf.children.length; x++) {

                var leaf_ch = leaf.children[x];
                children_ht += renderLeaf_(leaf_ch);
            }
        }

        return '<div class="leaf">' +
            renderBar_(leaf) +
            '<div class="children">' + children_ht + '</div>' +
            '</div>';
    };


    var set_ = function (new_tree) {
        tree_ = new_tree;
    };


    var get_ = function ( max_scale ) {

        var ht = renderLeaf_(tree_);
        var scale_ht = max_scale ? get_scale_ht_( max_scale ) : "";

        return ht + scale_ht;
    };


    var get_scale_ht_ = function(max_scale) {

        var increment = max_scale / 10;
        var scale_ht = '';
        var loc = 0;

        for (var i = 0; i <= 10; i++) {
            var x = Math.round(i * increment);

            if( i === 10 ) {
                x = "";
            }
            scale_ht += '<div class="scale" style="left: ' + loc + 'px;">' + x + '</div>';
            loc += 95;
        }

        return '<div class="scale_container">' + scale_ht + '</div>';
    };



    return {
        get: get_,
        set: set_
    }
}();
