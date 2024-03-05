$(document).ready(function(){
    $("#start-btn").click(function(){
        $.ajax({
            type: "POST",
            url: "/start_detection",
            success: function(data){
                $("#status").text(data.status);
                $("#start-btn").hide();
                $("#stop-btn").show();
            },
            error: function(){
                $("#status").text("Error: Unable to start detection.");
            }
        });
    });

    $("#stop-btn").click(function(){
        $.ajax({
            type: "POST",
            url: "/stop_detection",
            success: function(data){
                $("#status").text(data.status);
                $("#stop-btn").hide();
                $("#start-btn").show();
            },
            error: function(){
                $("#status").text("Error: Unable to stop detection.");
            }
        });
    });
});